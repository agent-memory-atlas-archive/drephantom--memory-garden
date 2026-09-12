'use strict';
// A bounded reading view over explicit links. Layout positions carry no semantic claims.
window.MemoryGardenGraph = (() => {
  const layouts = new Map(), ns = 'http://www.w3.org/2000/svg';
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const hash = value => { let n = 2166136261; for (const c of String(value)) n = Math.imul(n ^ c.codePointAt(0), 16777619); return n >>> 0; };
  const dateOf = item => item.source_kind === 'chat' && item.date_range ? item.date_range.start : item.event_time || item.recorded_at || '';
  function draw({container, nodes, edges, focusUid = '', scope = 'local', cacheKey = '', onNode, onEdge}) {
    const controller = new AbortController(), signal = controller.signal, global = scope === 'global';
    const key = scope + '|' + cacheKey, previous = layouts.get(key), reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const cached = previous && previous.positions.size === nodes.length && nodes.every(item => previous.positions.has(String(item.id))) ? previous : null;
    let stopped = false, frame = 0, resizeFrame = 0, drag = null, pan = null, suppressClick = false;
    let width = Math.max(280, container.clientWidth || 600), height = 400, worldW, worldH;
    let zoom = 1, cameraX = 0, cameraY = 0, initialized = false, panning = false, cameraTouched = false;
    let selected = new Set(), hovering = new Set(), activeIndex = 0;
    const html = (tag, cls, text) => { const el = document.createElement(tag); if (cls) el.className = cls; if (text !== undefined) el.textContent = text; return el; };
    const shape = (tag, attrs = {}, text) => { const el = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k, String(v))); if (text !== undefined) el.textContent = text; return el; };
    const listen = (el, type, callback, options = {}) => el.addEventListener(type, callback, {...options, signal});
    const button = (label, callback) => { const el = html('button', 'text-button', label); el.type = 'button'; listen(el, 'click', callback); return el; };
    const toolbar = html('div', 'graph-toolbar'), controls = html('div', 'graph-controls');
    const status = html('span', 'graph-zoom-status'); status.setAttribute('aria-live', 'off');
    const svg = shape('svg', {class:'graph-svg',role:'group','aria-label': global ? '全部记录关系图。方向键选择记录，Enter 查看原文。' : '局部记录关系图。方向键选择记录，Enter 查看原文。'});
    svg.style.touchAction = 'pan-y';
    const world = shape('g'); svg.append(world);
    const panButton = button('移动画布', () => {
      panning = !panning; panButton.setAttribute('aria-pressed', String(panning));
      svg.classList.toggle('is-panning', panning); svg.style.touchAction = panning ? 'none' : 'pan-y';
    });
    panButton.setAttribute('aria-pressed', 'false');
    controls.append(button('缩小', () => setZoom(zoom / 1.3)), status, button('放大', () => setZoom(zoom * 1.3)), button('适应全部', () => fit()), panButton, button('重新布局', () => reset()));
    toolbar.append(html('span', '', global ? '从一条记录走进它的联系' : '较早的记录 → 较近的记录'), controls);
    container.replaceChildren(toolbar, svg, html('p', 'graph-hint', '点选记录查看原文；拖动节点整理位置。方向键选择，Enter 打开。手机可开启“移动画布”后拖动背景。'));
    height = Math.max(300, svg.getBoundingClientRect().height || 400);
    width = Math.max(280, svg.getBoundingClientRect().width || width);
    worldW = global ? Math.max(width, Math.sqrt(nodes.length) * 118) : Math.max(width, 340);
    worldH = global ? Math.max(height, Math.sqrt(nodes.length) * 105) : Math.max(height, Math.ceil(nodes.length / 4) * 135);
    const bodies = [...nodes].sort((a,b) => String(a.id).localeCompare(String(b.id))).map((item,index) => ({item,index,id:String(item.id),x:0,y:0,ax:0,ay:0,vx:0,vy:0,pinned:false,nearest:Infinity}));
    const byId = new Map(bodies.map(body => [body.id, body]));
    const links = edges.map(edge => ({edge,from:byId.get(String(edge.source)),to:byId.get(String(edge.target)),bendSign:hash(edge.id)%2 ? 1 : -1})).filter(link => link.from && link.to);
    const initialTicks = global ? (bodies.length > 300 ? 36 : 65) : 85;
    const neighbors = new Map(bodies.map(body => [body.id, new Set()]));
    links.forEach(({from,to}) => { neighbors.get(from.id).add(to.id); neighbors.get(to.id).add(from.id); });
    function seed() {
      const dates = bodies.map(body => Date.parse(dateOf(body.item))).filter(Number.isFinite);
      const first = dates.length ? Math.min(...dates) : 0, last = dates.length ? Math.max(...dates) : 0;
      const components = [], visited = new Set();
      if (global) bodies.forEach(body => {
        if (visited.has(body.id)) return;
        const component = [body]; visited.add(body.id);
        for (let i = 0; i < component.length; i++) neighbors.get(component[i].id).forEach(id => { if (!visited.has(id)) { visited.add(id); component.push(byId.get(id)); } });
        components.push(component);
      });
      components.sort((a,b) => b.length - a.length);
      const boxes = new Map();
      function partition(groups, x, y, w, h) {
        if (!groups.length) return;
        if (groups.length === 1) { boxes.set(groups[0], {x,y,w,h}); return; }
        const total=groups.reduce((sum,group)=>sum+Math.max(2,group.length),0);
        let prefix=0,cut=1,leftWeight=0,best=Infinity;
        for (let i=0;i<groups.length-1;i++) { prefix+=Math.max(2,groups[i].length);const gap=Math.abs(prefix-total/2);if(gap<best){best=gap;cut=i+1;leftWeight=prefix;} }
        const ratio=leftWeight/total;
        if (w>=h) { partition(groups.slice(0,cut),x,y,w*ratio,h);partition(groups.slice(cut),x+w*ratio,y,w*(1-ratio),h); }
        else { partition(groups.slice(0,cut),x,y,w,h*ratio);partition(groups.slice(cut),x,y+h*ratio,w,h*(1-ratio)); }
      }
      partition(components,30,35,worldW-60,worldH-80);
      components.forEach(component => {
        const box=boxes.get(component),cx=box.x+box.w/2,cy=box.y+box.h/2;
        const radius = Math.min(box.w,box.h) * .43;
        component.forEach((body,i) => { const angle = i * 2.39996, r = component.length < 2 ? 0 : radius * Math.sqrt((i + 1) / component.length); body.x = cx + Math.cos(angle) * r; body.y = cy + Math.sin(angle) * r; });
      });
      bodies.forEach((body,index) => {
        if (!global) {
          const date = Date.parse(dateOf(body.item));
          body.x = bodies.length === 1 ? worldW / 2 : Number.isFinite(date) && last > first ? 75 + (date-first) / (last-first) * (worldW-150) : 75 + (hash(body.id) % 1000) / 1000 * (worldW-150);
          body.y = bodies.length === 1 ? worldH / 2 : 60 + ((index * 3 % bodies.length) + .5) / bodies.length * (worldH-135);
          // For multiples of three, a coprime stride avoids stacking every node on one row.
          if (bodies.length % 3 === 0) body.y = 60 + (index + .5) / bodies.length * (worldH-135);
        }
        body.ax = body.x; body.ay = body.y; body.vx = 0; body.vy = 0; body.pinned = false;
      });
    }
    seed();
    if (cached) {
      bodies.forEach(body => { const saved = cached.positions.get(body.id); if (saved) Object.assign(body, {x:saved.x,y:saved.y,ax:saved.x,ay:saved.y,pinned:saved.pinned}); });
      if (!global) reproject(cached.worldW, cached.worldH, worldW, worldH);
    }
    function measureNeighbors() {
      if (!global) return;
      bodies.forEach(body => { body.nearest=Infinity; });
      for (let i=0;i<bodies.length;i++) for (let j=i+1;j<bodies.length;j++) {
        const a=bodies[i],b=bodies[j],distance=Math.hypot(a.x-b.x,a.y-b.y);
        a.nearest=Math.min(a.nearest,distance);b.nearest=Math.min(b.nearest,distance);
      }
    }
    measureNeighbors();
    function labels() {
      const compact = bodies.length > (global ? 20 : 12) && zoom < .8;
      svg.classList.toggle('is-overview', compact);
      bodies.forEach(body => {
        const visible = !compact || selected.has(body.id) || hovering.has(body.id) || body.id === String(focusUid);
        body.title.style.visibility = body.date.style.visibility = visible ? 'visible' : 'hidden';
        body.title.style.pointerEvents = body.date.style.pointerEvents = compact ? 'none' : 'auto';
        const scale = Math.max(1, 1 / zoom);
        body.title.style.fontSize = 14 * scale + 'px'; body.date.style.fontSize = 11 * scale + 'px';
        body.title.style.strokeWidth = 4 * scale + 'px'; body.date.style.strokeWidth = 3 * scale + 'px';
        body.title.setAttribute('y', 30 * scale); body.date.setAttribute('y', 47 * scale);
        const dotRadius = Math.max(global ? 8 : 11, 3 / zoom);
        body.dot.setAttribute('r', global && compact ? Math.min(dotRadius,body.nearest*.35) : dotRadius);
        if (global) body.hit.setAttribute('r', compact ? Math.min(12 / zoom, body.nearest * .42) : 20 / Math.max(1, zoom));
      });
    }
    function highlight() {
      const ids = hovering.size ? hovering : selected, adjacent = new Set(ids);
      ids.forEach(id => neighbors.get(id)?.forEach(other => adjacent.add(other)));
      bodies.forEach(body => { body.group.classList.toggle('is-muted', ids.size > 0 && !adjacent.has(body.id)); body.group.classList.toggle('is-highlighted', ids.has(body.id)); });
      links.forEach(link => { const related = ids.has(link.from.id) || ids.has(link.to.id); link.path.classList.toggle('is-muted', ids.size > 0 && !related); link.path.classList.toggle('is-highlighted', related); });
      labels();
    }
    function camera() {
      svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      world.setAttribute('transform', 'translate(' + cameraX + ' ' + cameraY + ') scale(' + zoom + ')');
      status.textContent = Math.round(zoom * 100) + '%'; labels();
    }
    function setZoom(value) {
      cameraTouched = true;
      const next = clamp(value, .08, 3), x = width / 2, y = height / 2;
      cameraX = x - (x-cameraX) * next / zoom; cameraY = y - (y-cameraY) * next / zoom; zoom = next; camera();
    }
    function fit() {
      if (!bodies.length) return;
      const left = Math.min(...bodies.map(body => body.x)), right = Math.max(...bodies.map(body => body.x));
      const top = Math.min(...bodies.map(body => body.y)), bottom = Math.max(...bodies.map(body => body.y));
      // Labels retain screen-sized fonts when zoomed out, so their margins must be screen pixels too.
      const overview = global && bodies.length > 20 && !selected.size && !focusUid;
      const horizontalMargin = overview ? 44 : 160;
      zoom = clamp(Math.min((width-horizontalMargin)/Math.max(1,right-left), (height-105)/Math.max(1,bottom-top), 1), .08, 1);
      cameraX = width/2 - (left+right)/2*zoom; cameraY = (height-25)/2 - (top+bottom)/2*zoom; camera();
    }
    function point(event, coordinates = world) {
      const p = svg.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
      const matrix = coordinates.getScreenCTM(); return matrix ? p.matrixTransform(matrix.inverse()) : p;
    }
    function choose(body) {
      selected = new Set([body.id]); activeIndex = body.index; bodies.forEach(item => item.group.setAttribute('tabindex', item === body ? '0' : '-1'));
      if (body.x*zoom+cameraX < 80 || body.x*zoom+cameraX > width-80 || body.y*zoom+cameraY > height-65) { cameraX=width/2-body.x*zoom;cameraY=(height-25)/2-body.y*zoom;camera(); }
      highlight(); onNode?.(body.item);
    }
    links.forEach(link => {
      link.path = shape('path', {class:'graph-edge-line','aria-hidden':'true'});
      link.hit = shape('path', {class:'graph-edge-hit','aria-hidden':'true'});
      link.hit.append(shape('title', {}, link.from.item.title + ' → ' + link.to.item.title));
      listen(link.hit, 'pointerenter', () => { hovering = new Set([link.from.id,link.to.id]); highlight(); });
      listen(link.hit, 'pointerleave', () => { hovering.clear(); highlight(); });
      listen(link.hit, 'click', () => { if (suppressClick) { suppressClick = false; return; } selected = new Set([link.from.id,link.to.id]); highlight(); onEdge?.(link.edge,link.from.item,link.to.item); });
      world.append(link.path, link.hit);
    });
    bodies.forEach(body => {
      const group = shape('g', {class:'graph-node' + (!neighbors.get(body.id).size ? ' graph-isolated' : ''),tabindex:body.index ? -1 : 0,role:'button','aria-label':'查看记录：' + body.item.title});
      body.group = group; group.style.touchAction = 'none';
      body.hit = global ? shape('circle', {r:16,class:'graph-node-hit'}) : shape('rect', {x:-68,y:-25,width:136,height:83,rx:14,class:'graph-node-hit'});
      group.append(shape('title', {}, body.item.title), body.hit);
      body.dot = shape('circle', {r:global ? 8 : 11,class:body.item.matched_topic ? 'graph-dot matched' : 'graph-dot'});
      group.append(shape('circle', {r:24,class:'graph-node-halo','pointer-events':'none'}), body.dot);
      if (body.id === String(focusUid)) group.append(shape('circle', {r:17,class:'graph-focus-ring','pointer-events':'none'}));
      const chars = [...(body.item.title || '未命名记录')];
      body.title = shape('text', {x:0,y:30,'text-anchor':'middle',class:'graph-node-title'}, chars.length > 10 ? chars.slice(0,9).join('') + '…' : chars.join(''));
      body.date = shape('text', {x:0,y:47,'text-anchor':'middle',class:'graph-node-date'}, String(dateOf(body.item)).slice(0,10) || '日期待核对');
      group.append(body.title, body.date);
      listen(group, 'pointerenter', () => { hovering = new Set([body.id]); highlight(); });
      listen(group, 'pointerleave', () => { if (!drag) { hovering.clear(); highlight(); } });
      listen(group, 'focus', () => { activeIndex = body.index; hovering = new Set([body.id]); highlight(); });
      listen(group, 'blur', () => { hovering.clear(); highlight(); });
      listen(group, 'pointerdown', event => {
        if (event.button !== 0 || drag || pan) return;
        event.stopPropagation(); const p = point(event); suppressClick = false;
        drag = {body,pointer:event.pointerId,x:event.clientX,y:event.clientY,offsetX:p.x-body.x,offsetY:p.y-body.y};
        group.setPointerCapture(event.pointerId); group.classList.add('is-dragging');
      });
      listen(group, 'pointermove', event => {
        if (!drag || drag.body !== body || drag.pointer !== event.pointerId) return;
        if (Math.hypot(event.clientX-drag.x,event.clientY-drag.y) <= 4 && !suppressClick) return;
        suppressClick = true; const p = point(event); body.x = clamp(p.x-drag.offsetX,20,worldW-20); body.y = clamp(p.y-drag.offsetY,20,worldH-40);
        body.ax = body.x; body.ay = body.y; body.pinned = true; body.vx = body.vy = 0; render();
      });
      const release = event => {
        if (!drag || drag.body !== body || drag.pointer !== event.pointerId) return;
        drag = null; group.classList.remove('is-dragging'); if (group.hasPointerCapture(event.pointerId)) group.releasePointerCapture(event.pointerId);
        if (event.type === 'pointercancel') suppressClick = true;
        save(); if (suppressClick) { measureNeighbors();settle(Math.min(40,initialTicks)); } hovering.clear(); highlight();
      };
      listen(group, 'pointerup', release); listen(group, 'pointercancel', release);
      listen(group, 'click', () => { if (suppressClick) { suppressClick = false; return; } choose(body); });
      listen(group, 'keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(body); return; }
        const direction = {ArrowLeft:-1,ArrowUp:-1,ArrowRight:1,ArrowDown:1}[event.key];
        if (!direction && !['Home','End'].includes(event.key)) return;
        event.preventDefault(); activeIndex = event.key === 'Home' ? 0 : event.key === 'End' ? bodies.length-1 : (activeIndex+direction+bodies.length)%bodies.length;
        bodies.forEach(item => item.group.setAttribute('tabindex', item.index === activeIndex ? '0' : '-1'));
        const next = bodies[activeIndex];
        if (next.x*zoom+cameraX < 60 || next.x*zoom+cameraX > width-60 || next.y*zoom+cameraY < 45 || next.y*zoom+cameraY > height-65) { cameraX = width/2-next.x*zoom; cameraY = height/2-next.y*zoom; camera(); }
        next.group.focus({preventScroll:true});
      });
      world.append(group);
    });
    listen(svg, 'pointerdown', event => {
      suppressClick = false;
      if (event.button !== 0 || drag || pan || event.target.closest('.graph-node') || (event.pointerType === 'touch' && !panning)) return;
      const p = point(event,svg); pan = {pointer:event.pointerId,x:p.x,y:p.y,cameraX,cameraY}; svg.setPointerCapture(event.pointerId);
    });
    listen(svg, 'pointermove', event => {
      if (!pan || pan.pointer !== event.pointerId) return;
      const p = point(event,svg), dx = p.x-pan.x, dy = p.y-pan.y; if (Math.hypot(dx,dy)>4) suppressClick = true;
      if (suppressClick) cameraTouched = true;
      cameraX = pan.cameraX+dx; cameraY = pan.cameraY+dy; camera();
    });
    const endPan = event => { if (!pan || pan.pointer !== event.pointerId) return; pan = null; if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId); };
    listen(svg, 'pointerup', endPan); listen(svg, 'pointercancel', endPan);
    function render() {
      bodies.forEach(body => body.group.setAttribute('transform', 'translate(' + body.x.toFixed(2) + ' ' + body.y.toFixed(2) + ')'));
      links.forEach(link => {
        const dx = link.to.x-link.from.x, dy = link.to.y-link.from.y, distance = Math.hypot(dx,dy) || 1;
        const bend = link.bendSign * Math.min(25,distance*.1);
        const path = 'M ' + link.from.x + ' ' + link.from.y + ' Q ' + ((link.from.x+link.to.x)/2-dy/distance*bend) + ' ' + ((link.from.y+link.to.y)/2+dx/distance*bend) + ' ' + link.to.x + ' ' + link.to.y;
        link.path.setAttribute('d',path); link.hit.setAttribute('d',path);
      });
    }
    function physics() {
      const separationX = global ? 48 : 138, separationY = global ? 48 : 80;
      bodies.forEach(body => { body.vx += (body.ax-body.x)*.012; body.vy += (body.ay-body.y)*.01; });
      links.forEach(({from,to}) => { const dx=to.x-from.x,dy=to.y-from.y,d=Math.hypot(dx,dy)||1,f=(d-(global ? 105 : 155))*.002; from.vx+=dx/d*f;from.vy+=dy/d*f;to.vx-=dx/d*f;to.vy-=dy/d*f; });
      function repel(a,b) {
        const dx=b.x-a.x||.5,dy=b.y-a.y||.5;
        if (Math.abs(dx)<separationX && Math.abs(dy)<separationY) { const fx=(separationX-Math.abs(dx))*.025*Math.sign(dx),fy=(separationY-Math.abs(dy))*.04*Math.sign(dy);a.vx-=fx;b.vx+=fx;a.vy-=fy;b.vy+=fy; }
      }
      if (global && bodies.length > 80) {
        // Only adjacent fixed-size cells can overlap. No tree or external layout engine is needed.
        const cells=new Map();
        bodies.forEach(body => { body.cellX=Math.floor(body.x/separationX);body.cellY=Math.floor(body.y/separationY);const key=body.cellX+','+body.cellY;if(!cells.has(key))cells.set(key,[]);cells.get(key).push(body); });
        bodies.forEach(body => { for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(const other of cells.get((body.cellX+dx)+','+(body.cellY+dy))||[])if(other.index>body.index)repel(body,other); });
      } else for (let i=0;i<bodies.length;i++) for (let j=i+1;j<bodies.length;j++) repel(bodies[i],bodies[j]);
      let energy = 0;
      bodies.forEach(body => { if (body.pinned || drag?.body === body) { body.vx=body.vy=0;return; } body.vx=clamp(body.vx*.72,-4,4);body.vy=clamp(body.vy*.72,-4,4);body.x=clamp(body.x+body.vx,65,worldW-65);body.y=clamp(body.y+body.vy,35,worldH-60);energy+=Math.abs(body.vx)+Math.abs(body.vy); });
      return energy;
    }
    function save() {
      layouts.set(key, {worldW,worldH,positions:new Map(bodies.map(body => [body.id,{x:body.x,y:body.y,pinned:body.pinned}]))});
      if (layouts.size > 12) layouts.delete(layouts.keys().next().value);
    }
    function settle(maxTicks, fitAfter = false) {
      if (frame) cancelAnimationFrame(frame); let ticks = 0;
      if (reduced) { for (;ticks<maxTicks;ticks++) if (physics()<.08) break; render();measureNeighbors();if(fitAfter && !cameraTouched)fit();else labels();save();return; }
      const step = () => {
        if (stopped || !svg.isConnected) return;
        const energy=physics();render();ticks++;
        if (ticks<maxTicks && energy>.08) frame=requestAnimationFrame(step); else { frame=0;measureNeighbors();if(fitAfter && !cameraTouched)fit();else labels();save(); }
      };
      frame=requestAnimationFrame(step);
    }
    function reset() { cameraTouched=false;seed();measureNeighbors();selected.clear(); hovering.clear(); highlight(); render(); fit(); settle(initialTicks, true); }
    function reproject(oldW, oldH, nextW, nextH) {
      bodies.forEach(body => {
        for (const prop of ['x','ax']) body[prop]=clamp(65+(body[prop]-65)*(nextW-130)/Math.max(1,oldW-130),65,nextW-65);
        for (const prop of ['y','ay']) body[prop]=clamp(35+(body[prop]-35)*(nextH-95)/Math.max(1,oldH-95),35,nextH-60);
        body.vx=body.vy=0;
      });
    }
    function resize() {
      const rect = svg.getBoundingClientRect(); if (stopped || rect.width < 1 || rect.height < 1) return;
      const cx = (width/2-cameraX)/zoom, cy = (height/2-cameraY)/zoom;
      const changed = Math.abs(rect.width-width)>10 || Math.abs(rect.height-height)>10;
      width = rect.width; height = rect.height;
      if (!initialized) { initialized=true;fit(); }
      else if (!global && changed) {
        const nextW=Math.max(width,340),nextH=Math.max(height,Math.ceil(nodes.length/4)*135);
        reproject(worldW,worldH,nextW,nextH);worldW=nextW;worldH=nextH;
        cameraTouched=false;render();fit();settle(45,true);
      } else if (changed && !cameraTouched) fit();
      else { cameraX=width/2-cx*zoom;cameraY=height/2-cy*zoom;camera(); }
    }
    const observer = new ResizeObserver(() => { if (resizeFrame) cancelAnimationFrame(resizeFrame); resizeFrame=requestAnimationFrame(resize); });
    observer.observe(svg);
    listen(document, 'visibilitychange', () => { if (document.hidden && frame) { cancelAnimationFrame(frame);frame=0;measureNeighbors();labels();save(); } });
    render(); resize(); if (!cached) settle(initialTicks, true);
    return () => { if (stopped) return; stopped=true;controller.abort();observer.disconnect();if(frame)cancelAnimationFrame(frame);if(resizeFrame)cancelAnimationFrame(resizeFrame);save();drag=pan=null; };
  }
  return {draw};
})();
