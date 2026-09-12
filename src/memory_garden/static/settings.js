'use strict';
const form=document.getElementById('settings-form');
const field=name=>form.elements.namedItem(name);
let active=null;
async function request(url,body){
  const response=await window.MemoryGardenWorkspace.fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json().catch(()=>({}));
  if(!response.ok)throw new Error(data.error||'这次操作没有完成，请检查输入后重试。');
  return data;
}
function show(id,text,error=false){const el=document.getElementById(id);el.textContent=text;el.classList.toggle('error',error);}
function generationVisibility(){document.getElementById('generation-fields').hidden=field('backend').value==='local';}
field('backend').addEventListener('change',generationVisibility);
async function initialize(){
  document.getElementById('save').disabled=true;
  try{
    active=await request('/api/settings');
    for(const [key,value] of Object.entries(active)){const f=field(key);if(!f)continue;if(f.type==='checkbox')f.checked=!!value;else f.value=value;}
    for(const [key,flag] of [['llm_api_key','api_key_set'],['embedding_api_key','embedding_api_key_set'],['reranker_api_key','reranker_api_key_set']])field(key).placeholder=active[flag]?'已配置 · 留空不修改':'尚未配置';
    if(active.public_demo_mode){const note=document.getElementById('demo-note');note.hidden=false;note.textContent=active.demo_use_model?'这是 Agent 演示。对话和合成笔记片段会发送到已配置的生成模型；私人笔记库及旧对话不在此演示中。':'这是离线演示，使用模板对照。要体验模型理解与工具调用，可运行 start-agent-demo.bat。';}
    generationVisibility();document.getElementById('save').disabled=false;
  }catch{show('msg','未能读取当前设置。请确认本地服务正在运行。',true);}
}
form.addEventListener('submit',async e=>{
  e.preventDefault();const save=document.getElementById('save');save.disabled=true;show('msg','正在保存…');
  const body={};
  for(const f of form.elements){if(!f.name)continue;if(f.type==='checkbox')body[f.name]=f.checked;else if(f.type==='number')body[f.name]=Number(f.value);else if(f.type!=='password'||f.value.trim())body[f.name]=f.value.trim();}
  try{await request('/api/settings',body);show('msg','已保存。重启后生效，当前连接仍按原配置运行。');}
  catch(e){show('msg',e.message||'保存失败，请检查本地服务。',true);}
  finally{save.disabled=false;}
});
document.getElementById('sync').addEventListener('click',async e=>{const b=e.currentTarget;b.disabled=true;show('syncmsg','正在更新索引…');try{const r=await request('/api/sync',{});show('syncmsg',r.changed?'已更新，下一次回看会使用新记录。':'索引已是最新。');}catch(e){show('syncmsg',e.message,true);}finally{b.disabled=false;}});
document.getElementById('build').addEventListener('click',async e=>{
  const b=e.currentTarget;if(!active)return;
  const cloud=active.embedding_backend==='api';
  if(cloud&&!confirm('这会把当前笔记片段的标题、层级、标签和正文批量发送到 '+active.embedding_base_url+'。是否继续？'))return;
  b.disabled=true;show('buildmsg','正在构建…');
  try{const r=await request('/api/embeddings/build',{confirm_cloud_send:cloud});show('buildmsg','已更新 '+r.vectors_rebuilt+' 条向量缓存。');}
  catch(e){show('buildmsg',e.message,true);}finally{b.disabled=false;}
});
initialize();
