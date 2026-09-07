'use strict';
const $ = id => document.getElementById(id);
const assistantName = document.querySelector('.brand-name').textContent;
let threadId = null, busy = false;
const verdictLabels = {accurate:'已记下：贴近你的想法',partly_accurate:'已记下你的补充',no_change:'已记下：不构成变化',not_my_view:'已记下：这不是你的观点',insufficient_evidence:'已记下：证据还不够',defer:'先放在这里，不急着判断'};
function node(tag, cls, text) {const e=document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e;}
function button(text, action, cls='') {const b=node('button',cls,text);b.type='button';b.addEventListener('click',action);return b;}
async function api(url, body, signal) {
  let r;
  try {r=await fetch(url,{...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),signal});}
  catch(e){throw new Error(e.name==='AbortError'?'等候时间较长，结果可能仍在处理中。稍后可到历史里查看。':'暂时连不上本地花园。请检查服务是否还在运行，你写下的问题可以重试。');}
  const data=await r.json().catch(()=>({}));
  if(!r.ok)throw new Error(data.error || (typeof data.detail==='string'?data.detail:'这次操作没有完成，请稍后重试。'));
  return data;
}
function notice(text){$('notice').textContent=text;$('notice').hidden=!text;}
function nearBottom(){const m=$('main');return m.scrollHeight-m.scrollTop-m.clientHeight<140;}
function scrollBottom(){const m=$('main');m.scrollTop=m.scrollHeight;}
function showSource(c){
  $('source-title').textContent=c.title||'原始记录';
  const dates=[];if(c.event_time)dates.push('事件时间 '+c.event_time.slice(0,10));if(c.recorded_at)dates.push('记录时间 '+c.recorded_at.slice(0,10));
  $('source-dates').textContent=dates.join(' · ')||'这条记录没有明确日期';
  $('source-text').textContent=c.excerpt||'这次历史记录未保存片段。';
  $('source-path').textContent=(c.path||'')+(c.line_start?' · 第 '+c.line_start+' 行':'');
  $('source-dialog').showModal();
}
$('close-source').addEventListener('click',()=>$('source-dialog').close());
function renderReply(container,text,citations){
  const byId=new Map((citations||[]).map(c=>[Number(c.atom_id),c]));
  const pattern=/\[A(\d+)\]/g;let start=0,match;
  while((match=pattern.exec(text))!==null){
    container.append(document.createTextNode(text.slice(start,match.index)));
    const c=byId.get(Number(match[1]));
    if(c){const b=button('来源 '+match[1],()=>showSource(c),'inline-source');b.setAttribute('aria-label','查看来源 '+match[1]+'：'+c.title);container.append(b);}
    else container.append(document.createTextNode(match[0]));
    start=pattern.lastIndex;
  }
  container.append(document.createTextNode(text.slice(start)));
}
function addUser(text){const turn=node('article','turn user');turn.append(node('div','message-text',text));$('chat').append(turn);}
function feedback(turn,id,existing){
  const area=node('div','feedback');turn.append(area);
  if(existing){area.append(node('small','',verdictLabels[existing.verdict]||'你的反馈已保存'));return;}
  async function save(verdict,revision=''){
    area.querySelectorAll('button').forEach(b=>b.disabled=true);
    try{await api('/api/verdict',{message_id:id,verdict,user_revision:revision});area.replaceChildren(node('small','',verdictLabels[verdict]));}
    catch(e){area.querySelectorAll('button').forEach(b=>b.disabled=false);let err=area.querySelector('.error');if(!err){err=node('small','error');area.append(err);}err.textContent=e.message;}
  }
  area.append(button('这贴近我的想法',()=>save('accurate')),button('我想补充或修正',()=>{
    const editor=node('div','feedback-editor'),select=node('select'),input=node('textarea');
    select.setAttribute('aria-label','想修正哪一点');
    [['partly_accurate','有一部分需要补充'],['no_change','这不构成变化'],['not_my_view','这不是我的观点'],['insufficient_evidence','证据还不足以判断']].forEach(([v,t])=>{const o=node('option','',t);o.value=v;select.append(o);});
    input.placeholder='愿意的话，写下更贴近你的说法。也可以只选上面的一项。';input.setAttribute('aria-label','你的补充或修正');input.maxLength=3000;
    editor.append(select,input,button('记下这份修正',()=>save(select.value,input.value.trim())),button('暂时不写',()=>{area.remove();feedback(turn,id,null);},'text-button'));
    area.replaceChildren(editor);input.focus();
  }),button('暂时放着',()=>save('defer'),'text-button'));
}
function addAssistant(text,answer,id,backend,existing){
  const turn=node('article','turn assistant');turn.append(node('div','author',assistantName));
  const body=node('div','message-text');renderReply(body,text,answer&&answer.citations);turn.append(body);
  const tools=node('div','message-tools');
  tools.append(button('复制回答',async e=>{const b=e.currentTarget;try{await navigator.clipboard.writeText(text);b.textContent='已复制';}catch{notice('复制未完成。可以选中回答后手动复制。');}}));
  if(answer&&answer.citations&&answer.citations.length)tools.append(node('span','quiet',answer.citations.length+' 条原始记录可核对'));
  turn.append(tools);
  if(backend==='local_fallback')turn.append(node('p','quiet',answer&&answer.answer_type==='conversation'?'这次使用本地方式回应你的补充。':'这次模型未完成核对，这份回答采用本地记录对照。'));
  if(backend==='model_unavailable')turn.append(node('p','quiet','模型回应未完成，这条消息没有生成回溯结论。'));
  if(id&&answer&&['traced_change','no_clear_change'].includes(answer.answer_type))feedback(turn,id,existing);
  $('chat').append(turn);return turn;
}
async function submit(text){
  text=(text===undefined?$('q').value:text).trim();if(!text||busy)return;
  busy=true;$('send').disabled=true;$('chat').setAttribute('aria-busy','true');$('welcome').hidden=true;notice('');
  addUser(text);$('q').value='';$('q').style.height='auto';
  const pending=node('div','thinking','正在回应…');pending.setAttribute('role','status');$('chat').append(pending);scrollBottom();
  const slow=setTimeout(()=>pending.textContent='这次回应还需要一点时间。你可以先做别的事，完成后会保存在历史里。',12000);
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),240000);
  try{
    const data=await api('/api/ask',{question:text,thread_id:threadId},controller.signal);
    const stick=nearBottom();pending.remove();threadId=data.thread_id;history.replaceState(null,'','/?thread='+threadId);
    addAssistant(data.reply,data.answer,data.message_id,data.backend,null);if(stick)scrollBottom();
  }catch(e){pending.remove();const fail=node('div','notice');fail.append(node('div','',e.message),button('把问题放回输入框',()=>{$('q').value=text;$('q').focus();}));$('chat').append(fail);scrollBottom();}
  finally{clearTimeout(slow);clearTimeout(timeout);busy=false;$('send').disabled=false;$('chat').setAttribute('aria-busy','false');}
}
$('composer').addEventListener('submit',e=>{e.preventDefault();submit();});
$('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing&&e.keyCode!==229){e.preventDefault();submit();}});
$('q').addEventListener('input',()=>{$('q').style.height='auto';$('q').style.height=Math.min($('q').scrollHeight,140)+'px';});
document.querySelectorAll('[data-question]').forEach(b=>b.addEventListener('click',()=>submit(b.dataset.question)));
$('discover').addEventListener('click',async()=>{
  const b=$('discover');b.disabled=true;notice('');$('discoveries').replaceChildren(node('p','quiet','正在寻找可以放在一起回看的原话…'));
  try{
    const data=await api('/api/discover?limit=3');$('discoveries').replaceChildren();
    if(!data.candidates.length){$('discoveries').append(node('p','quiet','这次没有找到足够清楚的线索。没有明显变化，也是一种正常的状态。'));return;}
    data.candidates.forEach(c=>{
      const card=node('article','discovery');card.append(node('h2','',c.topic||'一条回看线索'));
      const compare=node('div','comparison');
      [[c.early_date,c.early_excerpt],[c.recent_date,c.recent_excerpt]].forEach(([date,excerpt])=>{const side=node('div');side.append(node('time','',date||'日期待核对'),node('p','',excerpt));compare.append(side);});
      card.append(compare,node('p','quiet','这是表达更具体了，还是想法真的变了？答案由你来判断。'));
      const actions=node('div','discovery-actions');
      actions.append(button('一起回看这条线索',()=>submit('关于'+c.topic+'，这些前后的表达构成变化吗？'),'primary'),button('这条暂时不看',async e=>{
        const b=e.currentTarget;b.disabled=true;
        try{await api('/api/discover/dismiss',{discovery_id:c.discovery_id});card.replaceChildren(node('p','quiet','先放下这条。接下来七天不会再主动展示它，也不会把这当作你对观点的判断。'));}
        catch(e){b.disabled=false;card.append(node('p','error',e.message));}
      },'text-button'));
      card.append(actions);$('discoveries').append(card);
    });
  }catch(e){$('discoveries').replaceChildren(node('p','error',e.message));}
  finally{b.disabled=false;}
});
async function initialize(){
  const wanted=new URLSearchParams(location.search).get('thread');
  if(wanted){busy=true;$('send').disabled=true;$('welcome').hidden=true;}
  try{
    const h=await api('/api/health');
    $('mode').textContent=h.public_demo_mode?(h.generation_connected?'Agent 演示 · '+h.generation_model+' · 合成记录':'离线演示 · 模板对照 · 合成记录'):(h.generation_connected?'Agent · '+h.generation_model:(h.cloud_retrieval?'本地回答 · 云端检索':'本地回看 · 无需联网'));
    if(!h.generation_connected&&h.backend!=='local')$('mode').textContent='Agent · 模型尚未配置';
    $('library').textContent='这里有 '+h.counts.sources+' 篇'+(h.public_demo_mode?'合成示例记录，先体验一次回看，再连接自己的笔记。':'已索引记录。更新笔记后，可以在设置里更新索引。');
    const link=node('a','','管理笔记与连接');link.href='/settings';$('library').append(link);
  }catch(e){$('mode').textContent='本地服务未连接';notice(e.message);}
  if(wanted){
    try{
      const messages=await api('/api/threads/'+encodeURIComponent(wanted));
      if(!Array.isArray(messages)||!messages.length)throw new Error('这段对话没有找到。可以开始一次新的回看。');
      threadId=Number(wanted);messages.forEach(m=>m.role==='user'?addUser(m.content):addAssistant(m.content,m.answer,m.id,null,m.verdict));scrollBottom();
    }catch(e){notice(e.message);$('welcome').hidden=false;history.replaceState(null,'','/');}
    finally{busy=false;$('send').disabled=false;}
  }
}
initialize();
