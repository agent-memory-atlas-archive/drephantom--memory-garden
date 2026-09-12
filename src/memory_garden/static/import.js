'use strict';
const $ = id => document.getElementById(id);
let preview = null, selectedFile = null, busy = false, fileVersion = 0, expiryTimer = null;
let importedOffset = 0, importedVersion = 0;
let localPage = 0;
const methods = ['file','paste','local'];
const panels = {file:'import-form',paste:'paste-form',local:'local-import'};
function node(tag, cls, text) { const element=document.createElement(tag);if(cls)element.className=cls;if(text!==undefined)element.textContent=text;return element; }
function notice(text) { $('import-notice').textContent=text;$('import-notice').hidden=!text; }
function ownNames() { return [...$('import-participants').querySelectorAll('input:checked')].map(input=>input.value); }
function commitAllowed() { return !!preview && (ownNames().length>0 || $('import-quotes-only').checked) && $('import-agent-access').checked; }
function setBusy(value) {
  busy=value;['preview-import','preview-paste','paste-content','chat-file','chat-format','chat-encoding','import-quotes-only','import-agent-access','method-local','method-file','method-paste','local-url','local-token','local-kind','local-count','connect-local','local-peer','local-more'].forEach(id=>$(id).disabled=value);
  $('preview-local').disabled=value || !$('local-peer').value;
  $('commit-import').disabled=value || !commitAllowed();$('import-participants').querySelectorAll('input').forEach(input=>input.disabled=value || input.dataset.unassignable==='true');
}
async function api(path, body) {
  const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),45000);
  try {
    const response=await window.MemoryGardenWorkspace.fetch(path,{...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),signal:controller.signal});
    const data=await response.json().catch(()=>({}));
    if(!response.ok)throw new Error(data.error || (typeof data.detail==='string'?data.detail:'这次操作没有完成，请核对格式后重试。'));
    return data;
  } catch(error) {
    if(error.name==='AbortError'||error instanceof TypeError)throw new Error('暂时连不上本地服务。确认导入后若连接中断，可以用同一预览重试；重复导入由服务端核对。');
    throw error;
  } finally { clearTimeout(timer); }
}
function clearPreview() {
  preview=null;clearTimeout(expiryTimer);$('import-preview').hidden=true;$('import-result').hidden=true;
  $('local-download-wrap').hidden=true;$('local-download').removeAttribute('href');
  $('import-quotes-only').checked=false;$('import-agent-access').checked=false;$('import-participants').replaceChildren();$('commit-import').disabled=true;
}
function refreshIdentity() {
  const names=ownNames();if(names.length)$('import-quotes-only').checked=false;
  $('commit-import').disabled=busy || !commitAllowed();
  $('import-own-count').textContent=names.length?'将 '+names.join('、')+' 的本人发言标为自己的记录；转发、引用和系统内容仍按引用保留。':$('import-quotes-only').checked?'全部消息将作为引用保留。':'尚未确认你的身份，暂不能导入。';
  $('import-messages').querySelectorAll('[data-sender]').forEach(card=>{
    const forceQuoted=card.dataset.forceQuoted==='true',own=names.includes(card.dataset.sender)&&!forceQuoted;
    card.classList.toggle('is-own',own);card.querySelector('.import-author-kind').textContent=own?'本人记录':forceQuoted?'转发 / 引用 / 系统内容':'他人的话';
  });
}
function renderPreview(data) {
  preview=data;$('import-preview').hidden=false;$('import-result').hidden=true;$('import-participants').replaceChildren();
  const counts=data.counts || {}, participants=data.participants||[];
  $('import-counts').textContent=(data.filename||selectedFile?.name||'聊天片段')+' · '+(counts.messages||0)+' 条消息 · '+(counts.participants||participants.length)+' 位说话人';
  participants.forEach(name=>{
    const label=node('label','import-person'), input=node('input');input.type='checkbox';input.value=name;
    if(name==='未知发言人'){input.disabled=true;input.dataset.unassignable='true';label.title='没有明确身份的消息只能作为引用保留。';}
    input.addEventListener('change',refreshIdentity);label.append(input,node('span','',name));$('import-participants').append(label);
  });
  $('import-warnings').replaceChildren();
  $('local-download-wrap').hidden=!data.connector;
  if(data.connector)$('local-download').href='/api/import/previews/'+encodeURIComponent(data.preview_id)+'/download?workspace='+encodeURIComponent(window.MemoryGardenWorkspace.id);
  (data.warnings||[]).forEach(warning=>$('import-warnings').append(node('p','notice',typeof warning==='string'?warning:JSON.stringify(warning))));
  if(data.preview_truncated)$('import-warnings').append(node('p','quiet','这里预览前 30 条消息。确认后会导入本文件解析出的全部 '+counts.messages+' 条消息，请同时核对原始导出文件。'));
  $('import-messages').replaceChildren();
  (data.messages||[]).forEach(message=>{
    const card=node('article','import-message');card.dataset.sender=message.sender;
    card.dataset.forceQuoted=String(message.force_quoted===true || message.can_be_own===false);
    const heading=node('div','import-message-heading');heading.append(node('strong','',message.sender||'说话人待核对'),node('span','import-author-kind','他人的话'));
    card.append(heading,node('time','quiet',message.timestamp||'时间待核对'),node('p','',message.text));
    if(message.text_truncated)card.append(node('small','quiet','这条正文较长，预览仅显示前 1,000 个字符，确认后导入完整正文。'));
    if(message.line_start)card.append(node('small','source-path','导出文件第 '+message.line_start+(message.line_end&&message.line_end!==message.line_start?'–'+message.line_end:'')+' 行'));
    $('import-messages').append(card);
  });
  const seconds=Number(data.expires_in_seconds)||900;
  $('import-expiry').textContent='预览将在 '+Math.ceil(seconds/60)+' 分钟后过期';
  clearTimeout(expiryTimer);expiryTimer=setTimeout(()=>{preview=null;$('commit-import').disabled=true;$('import-expiry').textContent='预览已过期，请回到上方重新预览。';},seconds*1000);
  refreshIdentity();$('import-preview').scrollIntoView({block:'start',behavior:'smooth'});
}
function showMethod(method, focus=false) {
  if(busy)return;
  clearPreview();notice('');$('paste-note').textContent='';
  methods.forEach(name=>{
    const chosen=name===method, tab=$('method-'+name);
    tab.classList.toggle('active',chosen);tab.setAttribute('aria-selected',String(chosen));tab.tabIndex=chosen?0:-1;
    $(panels[name]).hidden=!chosen;
    if(chosen&&focus)tab.focus();
  });
}
methods.forEach(method=>{
  const tab=$('method-'+method);tab.addEventListener('click',()=>showMethod(method));
  tab.addEventListener('keydown',event=>{
    if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){
      event.preventDefault();
      const next=event.key==='Home'?0:event.key==='End'?methods.length-1:(methods.indexOf(method)+(event.key==='ArrowRight'?1:-1)+methods.length)%methods.length;
      showMethod(methods[next],true);
    }
  });
});
function localConnection() { return {base_url:$('local-url').value.trim(),token:$('local-token').value.trim(),kind:$('local-kind').value}; }
function resetLocal() {
  localPage=0;$('local-peer').replaceChildren(node('option','','请选择会话'));$('local-peer').firstChild.value='';
  $('local-sessions').hidden=true;$('local-status').textContent='';$('preview-local').disabled=true;clearPreview();
}
['local-url','local-token'].forEach(id=>$(id).addEventListener('input',resetLocal));
$('local-kind').addEventListener('change',resetLocal);
$('local-count').addEventListener('change',clearPreview);
$('local-peer').addEventListener('change',()=>{clearPreview();$('preview-local').disabled=busy||!$('local-peer').value;});
async function loadLocalSessions(append=false) {
  if(busy)return;
  if(!append)resetLocal();notice('');setBusy(true);$('local-status').textContent='正在连接本机服务…';
  try{
    const data=await api('/api/import/local/sessions',{...localConnection(),page:append?localPage+1:1});
    (data.items||[]).forEach(item=>{const option=node('option','',item.name+' · '+item.identity);option.value=item.id;$('local-peer').append(option);});
    localPage=data.page;$('local-sessions').hidden=false;$('local-more').hidden=!data.has_more;
    $('local-status').textContent=(data.items||[]).length?'已连接，请选择一段会话':'这一页没有会话';
  }catch(error){$('local-status').textContent='连接未完成';notice(error.message);}
  finally{setBusy(false);}
}
$('connect-local').addEventListener('click',()=>loadLocalSessions());
$('local-more').addEventListener('click',()=>loadLocalSessions(true));
$('preview-local').addEventListener('click',async()=>{
  if(busy||!$('local-peer').value)return;
  clearPreview();notice('');setBusy(true);$('local-status').textContent='正在读取所选会话…';
  try{
    const data=await api('/api/import/local/preview',{...localConnection(),peer_id:$('local-peer').value,count:Number($('local-count').value)});
    renderPreview(data);$('local-status').textContent='已生成预览，尚未加入记忆';
  }catch(error){$('local-status').textContent='读取未完成';notice(error.message);}
  finally{setBusy(false);}
});
$('chat-file').addEventListener('change',()=>{
  fileVersion+=1;selectedFile=$('chat-file').files[0]||null;clearPreview();notice('');
  $('import-file-note').textContent=selectedFile?selectedFile.name+' · '+Math.ceil(selectedFile.size/1024)+' KiB':'';
});
$('chat-format').addEventListener('change',clearPreview);$('chat-encoding').addEventListener('change',clearPreview);
$('paste-content').addEventListener('input',()=>{clearPreview();$('paste-note').textContent='';});
$('paste-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;
  const content=$('paste-content').value;
  if(!content.trim()){notice('请先粘贴保留时间和发言人的聊天片段。');return;}
  if(new TextEncoder().encode(content).length>2*1024*1024){notice('片段超过 2 MiB，请分段预览。');return;}
  clearPreview();notice('');setBusy(true);$('paste-note').textContent='正在解析这段聊天…';
  try{
    const isJson=/^\s*[\[{]/.test(content)&&!/^\s*\[\d{4}[-/]/.test(content);
    const data=await api('/api/import/preview',{format:'wechat',filename:'微信聊天片段.'+(isJson?'json':'txt'),content,own_names:[]});
    renderPreview(data);$('paste-note').textContent='预览已生成，尚未加入记忆';
  }catch(error){notice(error.message);$('paste-note').textContent='预览未完成';}
  finally{setBusy(false);}
});
$('import-agent-access').addEventListener('change',refreshIdentity);
$('import-quotes-only').addEventListener('change',()=>{
  if($('import-quotes-only').checked)$('import-participants').querySelectorAll('input').forEach(input=>input.checked=false);refreshIdentity();
});
$('import-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;selectedFile=$('chat-file').files[0]||null;
  if(!selectedFile){notice('请先选择聊天导出文件。');return;}
  if(selectedFile.size>2*1024*1024){notice('文件超过 2 MiB。可以先按时间范围拆分导出，再分次导入。');return;}
  if(!/\.(txt|json|md)$/i.test(selectedFile.name)){notice('请选择 .txt、.json 或 .md 文本文件。');return;}
  const version=++fileVersion;clearPreview();notice('');setBusy(true);$('import-file-note').textContent='正在本地解析预览…';
  try {
    const bytes=await selectedFile.arrayBuffer();let content;
    try{content=new TextDecoder($('chat-encoding').value,{fatal:true}).decode(bytes);}catch{throw new Error('无法按所选编码读取文件。可以选择另一种导出编码，再核对预览是否出现乱码。');}
    if(new TextEncoder().encode(content).length>2*1024*1024)throw new Error('转换为 UTF-8 后的文本超过 2 MiB。请按时间范围拆分导出，再分次导入。');
    const data=await api('/api/import/preview',{format:$('chat-format').value,filename:selectedFile.name,content,own_names:[]});
    if(version!==fileVersion)return;renderPreview(data);$('import-file-note').textContent='预览已生成，尚未写入花园';
  }catch(error){notice(error.message);$('import-file-note').textContent='预览未完成';}
  finally{setBusy(false);}
});
$('commit-import').addEventListener('click',async()=>{
  if(busy||!commitAllowed())return;setBusy(true);notice('');
  try {
    const result=await api('/api/import/commit',{preview_id:preview.preview_id,own_names:ownNames(),confirm_agent_access:true});
    clearTimeout(expiryTimer);preview=null;$('import-preview').hidden=true;const box=$('import-result');box.hidden=false;box.replaceChildren();
    box.append(node('span','source-origin',result.duplicate?'这份记录已经在花园里':'已加入本地花园'),node('h2','',result.duplicate?'已找到相同的导入记录':'过去说过的话，也可以回看了。'));
    box.append(node('p','quiet',(result.indexed_messages||0)+' 条消息 · '+(result.own_messages||0)+' 条本人记录 · '+(result.quoted_messages||0)+' 条他人的话'));
    if(Number(result.superseded_imports)>0)box.append(node('p','quiet','已更新这份记录的身份归属。旧版本停止检索，历史副本仍保留。'));
    box.append(node('p','quiet','原始聊天导出文件与笔记库没有改动。后续连接模型回看时，仍遵循设置中的内容发送权限。'));
    const link=node('a','history-new','回到花园，开始回看 →');link.href='/';box.append(link);
    const activeMethod=methods.find(method=>$('method-'+method).getAttribute('aria-selected')==='true');
    $(activeMethod==='paste'?'paste-note':activeMethod==='local'?'local-status':'import-file-note').textContent='导入完成';
    box.scrollIntoView({block:'start',behavior:'smooth'});loadImports();
  }catch(error){notice(error.message);}
  finally{setBusy(false);}
});
async function loadImports(append=false) {
  append=append===true;const version=++importedVersion;
  if(!append){importedOffset=0;$('imported-list').replaceChildren(node('p','quiet','正在读取导入记录…'));}
  $('imports-more').disabled=true;
  try{
    const data=await api('/api/imports?limit=30&offset='+importedOffset);if(version!==importedVersion)return;
    if(!append)$('imported-list').replaceChildren();const items=data.items||[];
    $('imports-more').hidden=items.length<30;importedOffset+=items.length;
    if(!items.length&&importedOffset===0)$('imported-list').append(node('p','quiet','这里还没有导入的聊天。上方的预览不会自动写入记录。'));
    items.forEach(item=>{
      const card=node('article','imported-card'), active=!!item.is_present;
      const createdAt=String(item.created_at||''), parsedDate=new Date(createdAt);
      const importedAt=/(?:Z|[+-]\d{2}:\d{2})$/i.test(createdAt)&&!Number.isNaN(parsedDate.getTime())?parsedDate.toLocaleString('zh-CN',{hour12:false}):createdAt;
      const metadata=node('p','quiet',(importedAt?importedAt+' · ':'')+(item.indexed_messages||0)+' 条消息');
      if(createdAt)metadata.title=createdAt;
      card.append(node('span','source-origin',active?'可被 Agent 检索':'已停止检索'),node('h3','',item.title),metadata);
      const action=node('button','text-button',active?'停止检索此导入':'已停止检索');action.type='button';action.disabled=!active;
      action.addEventListener('click',async()=>{
        action.disabled=true;
        try{await api('/api/imports/'+encodeURIComponent(item.source_uid)+'/deactivate',{});card.querySelector('.source-origin').textContent='已停止检索';action.textContent='已停止检索';card.append(node('p','quiet','后续检索不再使用此导入。原导入副本与已有对话仍保留。'));}
        catch(error){action.disabled=false;card.append(node('p','error',error.message));}
      });
      card.append(action);$('imported-list').append(card);
    });
  }catch(error){if(version===importedVersion){if(!append)$('imported-list').replaceChildren();$('imported-list').append(node('p','error',error.message));}}
  finally{if(version===importedVersion)$('imports-more').disabled=false;}
}
$('refresh-imports').addEventListener('click',loadImports);$('imports-more').addEventListener('click',()=>loadImports(true));
loadImports();
