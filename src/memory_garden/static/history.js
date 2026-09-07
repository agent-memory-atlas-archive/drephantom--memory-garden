'use strict';
async function loadHistory(){
  const list=document.getElementById('history-list');
  try{
    const r=await fetch('/api/threads');if(!r.ok)throw new Error();const threads=await r.json();list.replaceChildren();
    if(!threads.length){const p=document.createElement('p');p.className='empty-state';p.textContent='这里还没有对话。从一个主题，或一条回看线索开始就好。';list.append(p);return;}
    threads.forEach(t=>{
      const a=document.createElement('a');a.className='history-item';a.href='/?thread='+encodeURIComponent(t.thread_id);
      const title=document.createElement('div');title.className='title';title.textContent=(t.title||'一段回看').slice(0,100);
      const meta=document.createElement('div');meta.className='meta';const date=t.last_at?new Date(t.last_at):null;
      meta.textContent=(date&&!Number.isNaN(date.getTime())?date.toLocaleString('zh-CN',{hour12:false}):'')+' · '+t.turns+' 条消息';a.append(title,meta);list.append(a);
    });
  }catch{list.textContent='暂时读不到历史。请确认本地服务正在运行，然后刷新页面。';list.className='notice';}
}
loadHistory();
