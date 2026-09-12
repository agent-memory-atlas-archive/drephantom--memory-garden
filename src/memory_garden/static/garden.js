'use strict';
const $ = id => document.getElementById(id);
const assistantName = document.querySelector('.brand-name').textContent;
const verdictLabels = {accurate:'贴近你的想法',partly_accurate:'有一部分需要补充',no_change:'这不构成变化',not_my_view:'这不是你的观点',insufficient_evidence:'证据还不足以判断',defer:'暂时不判断'};
const authorshipLabels = {user:'本人记录',quoted:'引用 / 摘录',ai_generated:'AI 生成内容',derived:'派生内容',current_turn:'这次对话的补充',user_verdict:'你的明确修正'};
let threadId = null, busy = false, currentView = 'chat', topic = '', workspaceId = null;
let mapData = null, mapTopic = null, mapVersion = 0, memoryVersion = 0, memoryOffset = 0;
let evidence = [], questionScope = [], questionTopic = '', pendingJob = null, pollTimer = null, pollErrors = 0;
let storageWarningShown = false, pollInProgress = false, lastMessageId = 0;
let graphCleanup = null;
let generationConnected = false;
let mapScope = 'local', mapFocus = '', mapQuery = '';
const threadQuestions = new Map();

function node(tag, cls, text) {
  const element = document.createElement(tag);
  if (cls) element.className = cls;
  if (text !== undefined) element.textContent = text;
  return element;
}
function button(text, action, cls = '') {
  const element = node('button', cls, text); element.type = 'button';
  element.addEventListener('click', action); return element;
}
async function api(url, body, timeoutMs = 20000) {
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await window.MemoryGardenWorkspace.fetch(url, {...(body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),signal:controller.signal});
  } catch { throw new Error('暂时连不上本地花园。问题和草稿仍留在这里，连接恢复后可以继续。'); }
  finally { clearTimeout(timeout); }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || (typeof data.detail === 'string' ? data.detail : '这次操作没有完成，请稍后重试。'));
    error.status = response.status; throw error;
  }
  return data;
}
function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
function localRead(key) { if (!key) return null; try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch { return null; } }
function localWrite(key, value) {
  if (!key) return false;
  try {
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    if (!storageWarningShown) {
      notice('浏览器未允许保存本地草稿。离开前请复制尚未发送的内容；已提交的回答仍会保存在服务端。');
      storageWarningShown = true;
    }
    return false;
  }
}
function draftKey() { return workspaceId ? 'mg:' + workspaceId + ':draft:' + (threadId || 'new') : null; }
function jobKey() { return workspaceId ? 'mg:' + workspaceId + ':job:' + threadId : null; }
function resizeInput() { $('q').style.height = 'auto'; $('q').style.height = Math.min($('q').scrollHeight, 140) + 'px'; }
function saveDraft() {
  localWrite(draftKey(), {text:$('q').value,scope:questionScope,topic:questionTopic});
  $('draft-status').textContent = $('q').value ? (workspaceId && !storageWarningShown ? '草稿已保存在此浏览器' : '草稿尚未保存，请等待连接恢复') : 'Enter 发送 · Shift + Enter 换行';
}
function restoreDraft() {
  if ($('q').value) { saveDraft(); return; }
  const saved = localRead(draftKey());
  if (!saved) return;
  $('q').value = typeof saved === 'string' ? saved : (saved.text || '');
  questionScope = Array.isArray(saved.scope) ? saved.scope.slice(0, 2) : [];
  questionTopic = typeof saved.topic === 'string' ? saved.topic.slice(0,120) : '';
  if (questionTopic) setTopic(questionTopic);
  renderScope(); resizeInput();
  if ($('q').value) $('draft-status').textContent = '已恢复这段对话的草稿';
}
function nearBottom() { const main = $('main'); return main.scrollHeight - main.scrollTop - main.clientHeight < 180; }
function scrollBottom() { const main = $('main'); main.scrollTop = main.scrollHeight; }
function setBusy(value) {
  busy = value; $('send').disabled = value || !workspaceId; $('chat').setAttribute('aria-busy', String(value));
  $('job-status').textContent = value ? '回答会继续生成，离开后可从历史回来' : generationConnected ? '你可以不同意，也可以暂时不判断。' : '当前可查找记录，连接模型后可展开讨论。';
}
function dateLabel(value, withTime = false) {
  if (!value) return '';
  if (!withTime) return String(value).slice(0, 10);
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', {hour12:false});
}
function sourceDates(citation) {
  const parts = [];
  if (citation.source_kind === 'chat' && citation.date_range) {
    const range = citation.date_range;
    parts.push('聊天 ' + range.start + (range.end !== range.start ? ' — ' + range.end : ''));
  }
  if (citation.event_time) parts.push('事件 ' + dateLabel(citation.event_time));
  if (citation.recorded_at) parts.push('记录 ' + dateLabel(citation.recorded_at));
  return parts.join(' · ') || '日期尚不明确';
}
function recordDate(item) { return item.source_kind === 'chat' && item.date_range ? item.date_range.start : item.event_time || item.recorded_at || ''; }
function sourceKey(citation) { return citation.atom_id != null ? 'atom:' + citation.atom_id : [citation.source_uid, citation.path, citation.excerpt].join('|'); }

// Only render plain text, paragraphs, lists, emphasis, code and known citations.
// Model text never enters an HTML interpreter.
function renderInline(container, text, citations) {
  const byId = new Map((citations || []).filter(c => c.atom_id != null).map(c => [Number(c.atom_id), c]));
  const pattern = /\[A(\d+)\]|\*\*([^*\n]+)\*\*|`([^`\n]+)`/g;
  let start = 0, match;
  while ((match = pattern.exec(text)) !== null) {
    container.append(document.createTextNode(text.slice(start, match.index)));
    if (match[1]) {
      const citation = byId.get(Number(match[1]));
      if (citation) {
        const sourceButton = button('来源 ' + match[1], () => showSource(citation), 'inline-source');
        sourceButton.setAttribute('aria-label', '查看来源 ' + match[1] + '：' + citation.title); container.append(sourceButton);
      } else container.append(document.createTextNode(match[0]));
    } else if (match[2]) {
      const strong = node('strong'); renderInline(strong, match[2], citations); container.append(strong);
    } else container.append(node('code', '', match[3]));
    start = pattern.lastIndex;
  }
  container.append(document.createTextNode(text.slice(start)));
}
function renderReply(container, text, citations) {
  String(text || '').split(/\n\s*\n/).forEach(block => {
    const lines = block.split('\n');
    if (lines.every(line => /^\s*[-*]\s+/.test(line))) {
      const list = node('ul');
      lines.forEach(line => { const item = node('li'); renderInline(item, line.replace(/^\s*[-*]\s+/, ''), citations); list.append(item); });
      container.append(list);
    } else {
      const heading = /^#{1,4}\s+/.test(block), paragraph = node(heading ? 'h3' : 'p');
      renderInline(paragraph, heading ? block.replace(/^#{1,4}\s+/, '') : block, citations); container.append(paragraph);
    }
  });
}
function addUser(text) { const turn = node('article', 'turn user'); turn.append(node('div', 'message-text', text)); $('chat').append(turn); }
function feedback(turn, id, existing, answerTopic) {
  const area = node('div', 'feedback'); turn.append(area);
  let saved = existing || null;
  function showSaved() {
    area.replaceChildren();
    if (saved) {
      const detail = node('div', 'feedback-saved');
      const state = saved.status === 'revoked' ? '已撤回，不再用于记忆' : saved.status === 'needs_review' ? '这份旧判断需要重新核对' : (verdictLabels[saved.verdict] || '已保存你的反馈');
      detail.append(node('strong', '', state));
      if (saved.user_revision) detail.append(node('p', 'feedback-revision', saved.user_revision));
      const label = saved.topic_label || saved.topic || answerTopic;
      detail.append(node('small', '', [label && !String(label).startsWith('terms:') ? label : '', dateLabel(saved.created_at, true)].filter(Boolean).join(' · ')));
      area.append(detail, button('重新判断', edit, 'text-button'));
    } else area.append(button('这贴近我的想法', () => save('accurate')), button('我想补充或修正', edit), button('暂时放着', () => save('defer'), 'text-button'));
  }
  async function save(verdict, revision = '') {
    area.querySelectorAll('button,select,textarea').forEach(element => element.disabled = true);
    try {
      saved = await api('/api/verdict', {message_id:id,verdict,user_revision:revision}); showSaved();
      if (currentView === 'memory') loadMemory();
    } catch (error) {
      area.querySelectorAll('button,select,textarea').forEach(element => element.disabled = false);
      let message = area.querySelector('.error');
      if (!message) { message = node('small', 'error'); area.append(message); } message.textContent = error.message;
    }
  }
  function edit() {
    const editor = node('div', 'feedback-editor'), select = node('select'), input = node('textarea');
    select.setAttribute('aria-label', '你现在的判断');
    Object.entries(verdictLabels).forEach(([value, text]) => { const option = node('option', '', text); option.value = value; select.append(option); });
    select.value = saved ? saved.verdict : 'partly_accurate'; input.value = saved ? (saved.user_revision || '') : '';
    input.placeholder = '愿意的话，写下更贴近你的说法。也可以只选择判断。'; input.setAttribute('aria-label', '你的补充或修正'); input.maxLength = 3000;
    const actions = node('div', 'inline-actions');
    actions.append(button('保存这次判断', () => save(select.value, input.value.trim()), 'primary'), button('取消', showSaved, 'text-button'));
    editor.append(select, input, node('small', 'quiet', '修正会替代这条回答的旧判断；原始记录始终保留。'), actions);
    area.replaceChildren(editor); select.focus();
  }
  showSaved();
}
function addAssistant(text, answer, id, backend, existing) {
  const turn = node('article', 'turn assistant'); if (id) turn.id = 'message-' + id;
  const author = node('div', 'author', assistantName);
  const origin = {local:'本地规则对照',local_dialogue:'本地回应',local_fallback:'本地规则回应',model_required:'模型未连接',model_unavailable:'模型回应未完成',budget_stop:'回应中断'}[backend];
  if (backend) author.append(node('span', 'answer-origin', origin || '模型回应'));
  turn.append(author);
  const body = node('div', 'message-text'); renderReply(body, text, answer && answer.citations); turn.append(body);
  const actions = node('div', 'message-tools');
  actions.append(button('复制回答', async event => {
    const action = event.currentTarget;
    try { await navigator.clipboard.writeText(text); action.textContent = '已复制'; }
    catch { notice('复制未完成。可以选中回答后手动复制。'); }
  }));
  if (['model_required','model_unavailable'].includes(backend)) {
    const connection = node('a', 'model-connection-link', backend === 'model_required' ? '连接对话模型' : '检查模型连接');
    connection.href = '/settings?' + new URLSearchParams({workspace:workspaceId}) + '#generation';
    actions.append(connection);
  }
  let references = null;
  if (answer && answer.citations && answer.citations.length) {
    if (answer.citations.length > 2) {
      references = node('div', 'reference-list'); references.hidden = true;
      answer.citations.forEach(citation => references.append(button((citation.atom_id ? '来源 ' + citation.atom_id + ' · ' : '') + citation.title, () => showSource(citation))));
      const reveal = button(answer.citations.length + ' 条记录可核对', () => { references.hidden = !references.hidden; reveal.setAttribute('aria-expanded', String(!references.hidden)); });
      reveal.setAttribute('aria-expanded', 'false'); actions.append(reveal);
    } else actions.append(button(answer.citations.length + ' 条记录可核对', () => {
      evidence = answer.citations.map(c => ({...c,reading_origin:'conversation'})); renderEvidence(true);
    }));
  }
  turn.append(actions);
  if (references) turn.append(references);
  if (backend === 'model_unavailable') turn.append(node('p', 'quiet', '模型回应未完成，这条消息没有生成回溯结论。'));
  if (id && answer && ['traced_change', 'no_clear_change'].includes(answer.answer_type)) feedback(turn, id, existing, answer.topic);
  $('chat').append(turn); return turn;
}
async function reloadThread(stick = false) {
  const messages = await api('/api/threads/' + threadId);
  if (!Array.isArray(messages)) throw new Error('这段对话暂时无法读取。');
  const scroll = $('main').scrollTop; $('chat').replaceChildren(); threadQuestions.clear(); let latestTopic = '';
  lastMessageId = messages.length ? Number(messages[messages.length - 1].id) : 0;
  messages.forEach(message => {
    if (message.role === 'user') { addUser(message.content); threadQuestions.set(Number(message.id), message.content); }
    else {
      addAssistant(message.content, message.answer, message.id, message.backend, message.verdict);
      if (message.answer && message.answer.topic) latestTopic = message.answer.topic;
    }
  });
  if (latestTopic && !questionTopic) setTopic(latestTopic);
  $('welcome').hidden = messages.length > 0;
  if (stick) scrollBottom(); else $('main').scrollTop = scroll;
  return messages;
}

function removePendingElement() { const old = $('pending-answer'); if (old) old.remove(); }
function showPending(text = '正在回应，你可以先做别的事…') {
  let pending = $('pending-answer');
  if (!pending) { pending = node('div', 'thinking'); pending.id = 'pending-answer'; pending.setAttribute('role', 'status'); $('chat').append(pending); }
  pending.replaceChildren(node('span', '', text)); return pending;
}
function persistJob() { if (threadId && pendingJob) localWrite(jobKey(), pendingJob); }
async function finishJob(job) {
  clearTimeout(pollTimer);
  const stick = currentView === 'chat' && nearBottom();
  // Reload persisted messages; never append the same completed reply twice.
  await reloadThread(stick); removePendingElement();
  localWrite(jobKey(), null); pendingJob = null; pollErrors = 0; setBusy(false);
  if (job.status === 'failed') {
    const failure = node('div', 'notice'); failure.append(node('p', '', job.error || '这次回应中断了，问题仍保存在对话中。'));
    if (job.question) failure.append(button('把问题放回草稿', () => restoreQuestion(job.question))); $('chat').append(failure);
  }
}
function restoreQuestion(text) {
  $('q').value = $('q').value.trim() ? $('q').value + '\n' + text : text;
  questionScope = []; questionTopic = ''; renderScope(); saveDraft(); resizeInput(); switchView('chat'); $('q').focus();
}
function interruptedPolling(error) {
  const pending = showPending(error.message + ' 已提交的请求不会自动重复发送。'); pending.className = 'notice';
  pending.append(button('重新连接这次请求', () => { pollErrors = 0; pollJob(true); }));
  $('job-status').textContent = '等候连接恢复 · 可从历史回到这段对话';
}
async function pollJob(recoverMissing = false) {
  clearTimeout(pollTimer); if (!pendingJob || pollInProgress) return;
  pollInProgress = true;
  try {
    let job;
    try { job = await api('/api/ask/jobs/' + encodeURIComponent(pendingJob.request_id)); }
    catch (error) {
      // A retry only reuses this request's idempotency key after confirmed absence.
      if (error.status !== 404 || !recoverMissing || !pendingJob.question) throw error;
      job = await api('/api/ask/jobs', {question:pendingJob.question,thread_id:threadId,request_id:pendingJob.request_id});
    }
    pollErrors = 0;
    if (['completed', 'failed'].includes(job.status)) { await finishJob({...job,question:pendingJob.question || job.question}); return; }
    pendingJob = {...pendingJob,...job}; persistJob();
    showPending(job.status === 'queued' ? '请求已保存，正在等候回应…' : '正在回应。离开页面后，仍可从历史回来继续。').className = 'thinking';
    pollTimer = setTimeout(() => pollJob(), 1800);
  } catch (error) {
    pollErrors += 1;
    if (pollErrors < 3) pollTimer = setTimeout(() => pollJob(recoverMissing), 3000); else interruptedPolling(error);
  } finally { pollInProgress = false; }
}
async function restoreJob() {
  if (!threadId || !workspaceId) return;
  const saved = localRead(jobKey()), response = await api('/api/threads/' + threadId + '/jobs');
  const items = Array.isArray(response) ? response : (response.items || []);
  const active = items.find(job => ['queued', 'running'].includes(job.status));
  const latestFailed = items.find(job => job.status === 'failed' && Number(job.user_message_id) >= lastMessageId);
  pendingJob = active ? {...(saved && saved.request_id === active.request_id ? saved : {}),...active} : (saved || latestFailed || null);
  if (pendingJob && !pendingJob.question) pendingJob.question = threadQuestions.get(Number(pendingJob.user_message_id));
  if (pendingJob) {
    setBusy(true); $('welcome').hidden = true; showPending('正在恢复这次回应…'); persistJob(); await pollJob(true);
  }
}
function scopedQuestion(text) {
  if (!questionScope.length) return questionTopic ? '关于「' + questionTopic + '」，' + text : text;
  const refs = questionScope.map(c => '《' + c.title + '》（' + sourceDates(c) + (c.atom_id ? '，来源 A' + c.atom_id : '') + '）').join('、');
  const scopedTopic = questionTopic || topic;
  return (scopedTopic ? '围绕「' + scopedTopic + '」，' : '') + '请查找并核对' + refs + '，再回答：\n' + text;
}
async function submit(text) {
  const suppliedQuestion = text !== undefined;
  text = (text === undefined ? $('q').value : text).trim(); if (!text || busy || !workspaceId) return;
  if (suppliedQuestion) { questionScope = []; questionTopic = ''; renderScope(); $('topic').value = topic; }
  else if ($('topic').value.trim() !== topic) setTopic($('topic').value, true);
  const question = scopedQuestion(text);
  if (question.length > 6000) { notice('加上选中的记录后，问题超过了长度限制。请缩短一些文字后再发送。'); return; }
  setBusy(true); notice(''); switchView('chat');
  try {
    if (!threadId) {
      const created = await api('/api/threads', {}); threadId = created.thread_id;
      history.replaceState(null, '', '/?' + new URLSearchParams({thread:String(threadId),workspace:workspaceId})); localWrite('mg:' + workspaceId + ':draft:new', null); saveDraft();
    }
    pendingJob = {request_id:crypto.randomUUID(),thread_id:threadId,question,status:'submitting'}; persistJob();
    addUser(question); $('welcome').hidden = true; showPending('正在保存这次请求…');
    $('q').value = ''; questionScope = []; questionTopic = ''; renderScope(); saveDraft(); resizeInput(); scrollBottom();
    const accepted = await api('/api/ask/jobs', {question,thread_id:threadId,request_id:pendingJob.request_id});
    pendingJob = {...pendingJob,...accepted}; persistJob(); await pollJob();
  } catch (error) {
    if (pendingJob) interruptedPolling(error); else { setBusy(false); notice(error.message); }
  }
}

function setTopic(value, fromUser = false) {
  const next = String(value || '').trim().slice(0, 120);
  if (next !== topic) { topic = next; mapData = null; mapTopic = null; mapFocus = ''; mapVersion += 1; memoryVersion += 1; }
  $('topic').value = topic;
  if (fromUser) { questionTopic = topic; questionScope = []; renderScope(); saveDraft(); }
}
function switchView(view, focus = false) {
  stopGraph(); mapVersion += 1;
  if (view !== 'chat' && $('topic').value.trim() !== topic) setTopic($('topic').value, true);
  currentView = view;
  document.querySelectorAll('[data-view]').forEach(tab => {
    const selected = tab.dataset.view === view;
    tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1; $('panel-' + tab.dataset.view).hidden = !selected;
    if (selected && focus) tab.focus();
  });
  if (['map', 'timeline'].includes(view)) loadMap(); if (view === 'memory') loadMemory();
}
function showSource(citation, origin = 'conversation') {
  if (!evidence.some(c => sourceKey(c) === sourceKey(citation))) {
    if (evidence.length === 2) evidence = [evidence[0]]; evidence.push({...citation,reading_origin:origin});
  }
  renderEvidence(true);
}
function renderEvidence(focus = false) {
  $('evidence-dock').hidden = !evidence.length; $('evidence-cards').replaceChildren();
  evidence.forEach((citation, index) => {
    const card = node('article', 'evidence-card'), top = node('div', 'evidence-card-top');
    top.append(node('span', 'source-origin', citation.reading_origin === 'map' ? '当前索引' : '引用时的片段'), button('移除', () => { evidence.splice(index, 1); renderEvidence(); }, 'text-button'));
    card.append(top, node('h3', '', citation.title || '原始记录'), node('p', 'quiet', sourceDates(citation)), node('p', 'source-authorship', authorshipLabels[citation.authorship] || '作者归属待核对'));
    if (citation.sender) card.append(node('p', 'source-authorship', '说话人：' + citation.sender));
    card.append(node('blockquote', '', citation.excerpt || '这条引用没有保存可阅读的片段。'));
    const lines = citation.line_start ? ' · 第 ' + citation.line_start + (citation.line_end && citation.line_end !== citation.line_start ? '–' + citation.line_end : '') + ' 行' : '';
    card.append(node('p', 'source-path', (citation.path || '来自对话中的表述') + lines)); $('evidence-cards').append(card);
  });
  $('evidence-cards').classList.toggle('single', evidence.length === 1);
  if (focus && evidence.length) $('evidence-dock').scrollIntoView({block:'nearest',behavior:'smooth'});
}
function renderScope() {
  const box = $('composer-scope'); box.replaceChildren(); box.hidden = !questionScope.length && !questionTopic;
  if (!box.hidden) box.append(node('span', '', questionScope.length ? '这次围绕：' + questionScope.map(c => c.title).join(' / ') : '这次围绕主题：' + questionTopic), button('取消限定', () => { questionScope = []; questionTopic = ''; renderScope(); saveDraft(); }, 'text-button'));
}
function emptyPanel(container, text) { container.replaceChildren(node('p', 'empty-state', text)); }
function stopGraph() { if (graphCleanup) { graphCleanup(); graphCleanup = null; } }
function syncMapControls() {
  $('map-local').setAttribute('aria-pressed', String(mapScope === 'local'));
  $('map-global').setAttribute('aria-pressed', String(mapScope === 'global'));
  $('map-search').hidden = mapScope !== 'global';
  $('map-clear-query').hidden = !mapQuery;
  $('map-focus').replaceChildren(); $('map-focus').hidden = !(mapScope === 'local' && mapFocus);
  if (!$('map-focus').hidden) $('map-focus').append(node('span', '', '正在围绕选中的记录展开'), button(topic ? '回到主题：' + topic : '取消记录焦点', () => setMapScope('local'), 'text-button'));
}
function setMapScope(scope, focusSource = '') {
  mapScope = scope; mapFocus = focusSource; syncMapControls();
  if (currentView !== 'map') switchView('map'); else loadMap();
}
async function loadMap(force = false) {
  const scope = currentView === 'map' ? mapScope : 'local';
  const focusSource = currentView === 'map' && scope === 'local' ? mapFocus : '';
  const key = JSON.stringify([scope, scope === 'global' ? mapQuery : topic, focusSource]);
  const version = ++mapVersion; stopGraph(); syncMapControls();
  if (scope === 'local' && !topic && !focusSource) {
    emptyPanel($('timeline'), '先在上方输入一个主题，或从对话中的主题开始。'); emptyPanel($('map-canvas'), '先选一个想回看的主题。图只展示与它有关的一小片记录。');
    $('map-canvas').append(button('先浏览全局图', () => setMapScope('global')));
    $('map-node-list').replaceChildren(); $('map-edges').replaceChildren(); $('map-meta').textContent = ''; $('map-selection').replaceChildren(); return;
  }
  if (!force && mapData && mapTopic === key) { renderMap(mapData); return; }
  if (currentView === 'timeline') emptyPanel($('timeline'), '正在整理相关记录…');
  else { emptyPanel($('map-canvas'), scope === 'global' ? '正在整理整个花园…' : '正在整理局部关系…'); $('map-meta').textContent = '正在读取…'; ['map-node-list','map-edges','map-selection'].forEach(id=>$(id).replaceChildren()); }
  try {
    const query = new URLSearchParams(scope === 'global' ? {scope,query:mapQuery,limit:'500'} : {scope,topic,limit:'40'}); if (focusSource) query.set('source_uid', focusSource);
    const data = await api('/api/map?' + query); if (version !== mapVersion) return;
    mapData = data; mapTopic = key; renderMap(data);
  } catch (error) {
    if (version !== mapVersion) return;
    stopGraph(); $('map-meta').textContent = '';
    emptyPanel($('timeline'), error.message); emptyPanel($('map-canvas'), error.message); $('map-canvas').append(button('重新读取', () => loadMap(true)));
  }
}
function nodeCitation(item) {
  return (item.citations || [])[0] || {source_uid:item.uid || item.id,title:item.title,path:item.path,event_time:item.event_time,recorded_at:item.recorded_at,authorship:item.authorship};
}
function openMapNode(item) {
  showSource(nodeCitation(item), 'map');
  $('map-selection').replaceChildren(node('h3', '', item.title), node('p', 'quiet', sourceDates(item)));
  $('map-selection').append(button('以这条记录展开局部图', () => setMapScope('local', item.uid || item.id)));
  if ((item.citations || []).length > 1) {
    const list = node('div', 'inline-actions');
    item.citations.forEach((citation, index) => list.append(button('片段 ' + (index + 1), () => showSource(citation, 'map')))); $('map-selection').append(list);
  }
}
function renderMap(data) {
  const nodes = data.nodes || [], edges = data.edges || [], meta = data.meta || {}, byId = new Map(nodes.map(item => [String(item.id), item]));
  const ordered = [...nodes].sort((a, b) => (recordDate(a) || '9999').localeCompare(recordDate(b) || '9999') || String(a.id).localeCompare(String(b.id)));
  if (currentView === 'timeline') {
    $('timeline').replaceChildren();
    if (!nodes.length) { emptyPanel($('timeline'), '这个主题还没有找到可展示的记录。可以换一个词，或在对话中说明你想回看的事情。'); return; }
    if (meta.omitted_nodes) $('timeline').append(node('p', 'quiet', '展示 ' + nodes.length + ' / ' + meta.total_candidates + ' 条相关记录，可以细化主题继续查看。'));
    ordered.forEach(item => {
    const citation = nodeCitation(item), entry = node('article', 'timeline-entry'), content = node('div', 'timeline-card');
    entry.append(node('time', '', dateLabel(recordDate(item)) || '日期待核对'));
    const dateKind = item.source_kind === 'chat' && item.date_range ? sourceDates(item) : item.event_time ? '按事件时间排列' : item.recorded_at ? '按记录时间排列' : '没有明确日期';
    const owner = item.source_kind === 'chat' ? '聊天记录 · 逐条保留说话人' : authorshipLabels[item.authorship] || '作者待核对';
    content.append(button(item.title, () => openMapNode(item), 'timeline-title'), node('p', 'quiet', dateKind + ' · ' + owner), node('p', 'timeline-excerpt', citation.excerpt || '选择记录查看来源信息。'));
    if (citation.sender) content.append(node('p', 'quiet', citation.sender + ' · ' + (authorshipLabels[citation.authorship] || '作者待核对')));
    content.append(button('留在手边核对', () => showSource(citation, 'map'), 'text-button')); entry.append(content); $('timeline').append(entry);
    });
    return;
  }
  if (currentView !== 'map') return;
  ['map-canvas','map-node-list','map-edges','map-selection'].forEach(id => $(id).replaceChildren());
  const global = data.scope === 'global', focus = byId.get(String(data.focus_source_uid));
  const counts = global ? ['当前可检索的记忆库', data.query ? '筛选结果 ' + nodes.length + ' / ' + meta.total_candidates + ' 条（库内共 ' + meta.total_visible + ' 条）' : '展示 ' + nodes.length + ' / ' + meta.total_visible + ' 条记录'] : [focus ? '以《' + focus.title + '》为中心' : '主题附近', '展示 ' + nodes.length + ' / ' + meta.total_candidates + ' 条记录'];
  counts.push(edges.length + ' 条明确链接');
  if (meta.omitted_nodes) counts.push('另有 ' + meta.omitted_nodes + ' 条未展示，可通过' + (global ? '搜索' : '细化主题') + '查看');
  if (meta.omitted_edges) counts.push('省略 ' + meta.omitted_edges + ' 条链接');
  if (meta.link_scan_truncated) counts.push('部分记录超过每条 ' + meta.links_per_source_limit + ' 个链接的扫描上限，连线并非全部');
  if (meta.unresolved_links) counts.push('部分链接尚未解析');
  $('map-meta').textContent = counts.join(' · ');
  if (!nodes.length) {
    emptyPanel($('map-canvas'), global ? (data.query ? '没有匹配这次筛选的记录。可以换一个词，或清除筛选查看整个花园。' : '记忆库里还没有可检索的记录。添加笔记或导入聊天后，可以在这里浏览。') : '还没有找到这个主题的记录。可以换一个词，或切换到全局图。');
    return;
  }
  ordered.forEach(item => $('map-node-list').append(button(item.title, () => openMapNode(item), 'map-node-chip')));
  drawGraph(nodes, edges, data.focus_source_uid, data.scope);
  if (!edges.length) emptyPanel($('map-edges'), '这些记录中还没有可解析的双链。记录出现在同一主题下，并不代表它们之间存在因果或观点变化。');
  edges.forEach(edge => {
    const from = byId.get(String(edge.source)), to = byId.get(String(edge.target)); if (!from || !to) return;
    const action = button(from.title + ' → ' + to.title, () => selectEdge(edge, from, to), 'map-edge');
    action.append(node('small', '', '笔记中的明确链接 · 查看依据')); $('map-edges').append(action);
  });
}
function selectEdge(edge, from, to) {
  const proof = edge.evidence || {}, panel = $('map-selection');
  panel.replaceChildren(node('span', 'source-origin', '笔记中的明确链接'), node('h3', '', from.title + ' → ' + to.title), node('blockquote', '', proof.quote || proof.excerpt || '这条链接未附带可展示的原文。'));
  panel.append(node('p', 'source-path', (proof.path || from.path || '') + (proof.line_start ? ' · 原文所在片段，第 ' + proof.line_start + (proof.line_end ? '–' + proof.line_end : '') + ' 行' : '')));
  panel.append(node('p', 'quiet', '这条线表示原笔记引用了另一条记录。它不表示因果，也不是系统对观点变化的判断。'));
  panel.append(button('并排核对两条记录', () => { evidence = [nodeCitation(from), nodeCitation(to)].map(c => ({...c,reading_origin:'map'})); renderEvidence(true); }));
  panel.scrollIntoView({block:'nearest',behavior:'smooth'});
}
function drawGraph(nodes, edges, focusUid, scope) {
  stopGraph();
  graphCleanup = window.MemoryGardenGraph.draw({
    container:$('map-canvas'), nodes, edges, focusUid, scope,
    cacheKey:mapTopic, onNode:openMapNode, onEdge:selectEdge,
  });
}

async function loadMemory(append = false) {
  append = append === true;
  const version = ++memoryVersion;
  if (!append) { memoryOffset = 0; emptyPanel($('memory-list'), '正在读取你确认过的记忆…'); }
  $('memory-more').disabled = true;
  try {
    const params = new URLSearchParams({topic,include_inactive:String($('memory-inactive').checked),limit:'30',offset:String(memoryOffset)}), data = await api('/api/memory?' + params);
    if (version !== memoryVersion) return;
    const items = data.items || [];
    if (!append) $('memory-list').replaceChildren();
    $('memory-more').hidden = items.length < 30;
    if (!items.length && memoryOffset === 0) { emptyPanel($('memory-list'), topic ? '这个主题还没有符合条件的长期记忆。你对回溯回答的明确确认或修正，会出现在这里。' : '这里还没有确认过的长期记忆。你可以先回看一个主题，再决定哪些理解值得留下。'); return; }
    memoryOffset += items.length;
    items.forEach(item => {
      const card = node('article', 'memory-card'), active = ['active','confirmed'].includes(item.status);
      const statuses = {active:'可用于回忆',confirmed:'可用于回忆',revoked:'已撤回',superseded:'已被后续修正替代',needs_review:'待重新核对'};
      const status = active && item.available_for_recall === false ? '暂不用于回忆 · 来源已移除或排除' : (statuses[item.status] || item.status);
      const kindNames = {user_revision:'你的补充或修正',confirmed_interpretation:'你确认贴近的解释',rejected_interpretation:'你否认的解释',uncertainty:'你认为仍需核对的解释'};
      const decision = item.evidence && verdictLabels[item.evidence.verdict];
      card.append(node('span', 'source-origin', status), node('h3', '', item.topic || '未分类主题'));
      card.append(node('p', 'memory-kind', (kindNames[item.kind] || '你留下的判断') + (decision ? ' · ' + decision : '')));
      if (item.kind === 'rejected_interpretation') card.append(node('p', 'quiet', '下面保留的是被你否认的解释，不代表你的观点。'));
      card.append(node('p', 'memory-statement', item.statement), node('p', 'quiet', dateLabel(item.created_at, true) + ' · 来自你的明确反馈'));
      const actions = node('div', 'inline-actions');
      if (item.source_thread_id) {
        const link = node('a', 'memory-source-link', '查看来源对话 / 修改判断');
        link.href = '/?' + new URLSearchParams({thread:String(item.source_thread_id),workspace:workspaceId}) + (item.source_message_id ? '#message-' + encodeURIComponent(item.source_message_id) : ''); actions.append(link);
      }
      if (active) actions.append(button('撤回这条记忆', async event => {
        const action = event.currentTarget; action.disabled = true;
        try {
          await api('/api/memory/' + item.id + '/revoke', {}); await loadMemory();
          if (threadId && Number(item.source_thread_id) === Number(threadId)) await reloadThread(false);
        }
        catch (error) { action.disabled = false; card.append(node('p', 'error', error.message)); }
      }, 'text-button'));
      card.append(actions, node('small', 'quiet', '撤回后不再用于 Agent 回忆，原对话与反馈记录仍保留。')); $('memory-list').append(card);
    });
  } catch (error) {
    if (version === memoryVersion) {
      if (append) $('memory-list').append(node('p', 'error', error.message));
      else emptyPanel($('memory-list'), error.message);
    }
  } finally { if (version === memoryVersion) $('memory-more').disabled = false; }
}

$('composer').addEventListener('submit', event => { event.preventDefault(); submit(); });
$('q').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); submit(); } });
$('q').addEventListener('input', () => { resizeInput(); saveDraft(); });
$('topic-form').addEventListener('submit', event => { event.preventDefault(); setTopic($('topic').value, true); switchView('timeline'); });
document.querySelectorAll('[data-view]').forEach(tab => {
  tab.addEventListener('click', () => switchView(tab.dataset.view));
  tab.addEventListener('keydown', event => {
    const tabs = [...document.querySelectorAll('[data-view]')], index = tabs.indexOf(tab); let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault(); switchView(tabs[next].dataset.view, true);
  });
});
document.querySelectorAll('.refresh-map').forEach(element => element.addEventListener('click', () => loadMap(true)));
$('map-local').addEventListener('click', () => setMapScope('local'));
$('map-global').addEventListener('click', () => setMapScope('global'));
$('map-search').addEventListener('submit', event => { event.preventDefault(); mapQuery = $('map-query').value.trim().slice(0,120); loadMap(); });
$('map-clear-query').addEventListener('click', () => { mapQuery = ''; $('map-query').value = ''; loadMap(); });
document.querySelectorAll('[data-question]').forEach(element => element.addEventListener('click', () => submit(element.dataset.question)));
$('refresh-memory').addEventListener('click', loadMemory); $('memory-inactive').addEventListener('change', loadMemory);
$('memory-more').addEventListener('click', () => loadMemory(true));
$('clear-evidence').addEventListener('click', () => { evidence = []; renderEvidence(); });
$('ask-evidence').addEventListener('click', () => {
  questionScope = evidence.map(c => ({title:c.title,atom_id:c.atom_id,event_time:c.event_time,recorded_at:c.recorded_at})); questionTopic = topic; renderScope(); saveDraft(); switchView('chat'); $('q').focus();
});
document.addEventListener('visibilitychange', () => { if (!document.hidden && pendingJob && pollErrors >= 3) { pollErrors = 0; pollJob(true); } });
window.addEventListener('pagehide', () => { saveDraft(); stopGraph(); });
document.addEventListener('workspace:before-switch', saveDraft);
$('discover').addEventListener('click', async () => {
  const action = $('discover'); action.disabled = true; notice(''); $('discoveries').replaceChildren(node('p', 'quiet', '正在寻找可以放在一起回看的原话…'));
  try {
    const data = await api('/api/discover?limit=3', undefined, 240000); $('discoveries').replaceChildren();
    if (!data.candidates.length) { $('discoveries').append(node('p', 'quiet', '这次没有找到足够清楚的线索。没有明显变化，也是一种正常的状态。')); return; }
    data.candidates.forEach(candidate => {
      const card = node('article', 'discovery'), comparison = node('div', 'comparison'); card.append(node('h2', '', candidate.topic || '一条回看线索'));
      [[candidate.early_date,candidate.early_excerpt],[candidate.recent_date,candidate.recent_excerpt]].forEach(([date, excerpt]) => { const side = node('div'); side.append(node('time', '', date || '日期待核对'), node('p', '', excerpt)); comparison.append(side); });
      card.append(comparison, node('p', 'quiet', '这是表达更具体了，还是想法真的变了？答案由你来判断。'));
      const actions = node('div', 'discovery-actions');
      actions.append(button('一起回看', () => submit('关于' + candidate.topic + '，这些前后的表达构成变化吗？'), 'primary'), button('先看时间线', () => { setTopic(candidate.topic); switchView('timeline'); }), button('这条暂时不看', async event => {
        const dismiss = event.currentTarget; dismiss.disabled = true;
        try { await api('/api/discover/dismiss', {discovery_id:candidate.discovery_id}); card.replaceChildren(node('p', 'quiet', '先放下这条。接下来七天不会再主动展示，也不会把这当作你对观点的判断。')); }
        catch (error) { dismiss.disabled = false; card.append(node('p', 'error', error.message)); }
      }, 'text-button')); card.append(actions); $('discoveries').append(card);
    });
  } catch (error) { $('discoveries').replaceChildren(node('p', 'error', error.message)); }
  finally { action.disabled = false; }
});
async function initialize() {
  const wanted = new URLSearchParams(location.search).get('thread'); setBusy(true);
  if (wanted && /^\d+$/.test(wanted)) { threadId = Number(wanted); $('welcome').hidden = true; }
  try {
    const health = await api('/api/health');
    if (!health.workspace_id) throw new Error('当前服务尚未返回工作区标识。请重启服务后刷新页面，再恢复草稿和请求。');
    workspaceId = health.workspace_id; restoreDraft();
    generationConnected = !!health.generation_connected;
    const hasImports = Number(health.imported_source_count) > 0, demoData = hasImports ? '含导入记录' : '合成记录';
    $('mode').textContent = health.public_demo_mode ? (generationConnected ? 'Agent 演示 · ' + health.generation_model + ' · ' + demoData : '模型未连接 · ' + demoData) : (generationConnected ? 'Agent · ' + health.generation_model : (health.cloud_retrieval ? '模型未连接 · 云端检索' : '模型未连接 · 本地检索'));
    if (!health.generation_connected && health.backend !== 'local') $('mode').textContent = 'Agent · 模型尚未配置' + (hasImports ? ' · 含导入记录' : '');
    $('connect-model').hidden = generationConnected;
    $('connect-model').href = '/settings?' + new URLSearchParams({workspace:workspaceId}) + '#generation';
    const libraryDescription = health.public_demo_mode ? (hasImports ? '已索引记录。导入记录与合成示例分别保留，可在导入页管理。' : '合成示例记录，先体验一次回看，再连接自己的笔记。') : '已索引记录。更新笔记后，可以在设置里更新索引。';
    $('library').textContent = '这里有 ' + health.counts.sources + ' 篇' + libraryDescription;
    const link = node('a', '', '管理笔记与连接'); link.href = '/settings'; $('library').append(link);
  } catch (error) { $('mode').textContent = '本地服务未连接'; notice(error.message); }
  if (threadId) {
    try {
      await reloadThread(true); await restoreJob();
      if (/^#message-\d+$/.test(location.hash)) { const target = $(location.hash.slice(1)); if (target) { target.classList.add('message-highlight'); target.scrollIntoView({block:'center'}); } }
    } catch (error) { notice(error.message + ' 刷新页面后会继续读取这段对话。'); }
  }
  if (!pendingJob) setBusy(false);
  if (workspaceId && new URLSearchParams(location.search).get('view') === 'map') { mapScope = new URLSearchParams(location.search).get('scope') === 'global' ? 'global' : 'local'; switchView('map'); }
}
initialize();
