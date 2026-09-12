'use strict';
const historyList = document.getElementById('history-list'), moreButton = document.getElementById('history-more'), historyStatus = document.getElementById('history-status');
let offset = 0, query = '', loading = false, generation = 0;
const pageSize = 30;
async function loadHistory(reset = false) {
  if (loading && !reset) return;
  const requestGeneration = reset ? ++generation : generation;
  if (reset) { offset = 0; historyList.replaceChildren(); }
  loading = true; moreButton.disabled = true; historyStatus.textContent = '正在读取…';
  try {
    const params = new URLSearchParams({q:query,limit:String(pageSize),offset:String(offset)}), response = await window.MemoryGardenWorkspace.fetch('/api/threads?' + params);
    if (!response.ok) throw new Error();
    const threads = await response.json(); if (requestGeneration !== generation) return;
    if (!threads.length && offset === 0) {
      const empty = document.createElement('p'); empty.className = 'empty-state'; empty.textContent = query ? '没有找到符合搜索词的对话。可以试试主题里的另一个词。' : '这里还没有对话。从一个主题，或一条回看线索开始就好。'; historyList.append(empty);
    }
    threads.forEach(thread => {
      const link = document.createElement('a'); link.className = 'history-item'; link.href = '/?' + new URLSearchParams({thread:String(thread.thread_id),workspace:window.MemoryGardenWorkspace.id});
      const title = document.createElement('div'); title.className = 'title'; title.textContent = (thread.title || '一段回看').slice(0, 120);
      const meta = document.createElement('div'); meta.className = 'meta'; const date = thread.last_at ? new Date(thread.last_at) : null;
      meta.textContent = (date && !Number.isNaN(date.getTime()) ? date.toLocaleString('zh-CN', {hour12:false}) + ' · ' : '') + thread.turns + ' 条消息';
      if (thread.pending || ['queued','running'].includes(thread.job_status)) meta.textContent += ' · 正在回应';
      link.append(title, meta); historyList.append(link);
    });
    offset += threads.length; moreButton.hidden = threads.length < pageSize;
    historyStatus.textContent = offset ? '已显示 ' + offset + ' 段对话' + (threads.length < pageSize ? ' · 已到末尾' : '') : '';
  } catch {
    if (requestGeneration === generation) { historyStatus.textContent = '暂时读不到历史。请确认本地服务正在运行，再点击重试。'; moreButton.hidden = false; moreButton.textContent = '重试读取'; }
  } finally { if (requestGeneration === generation) { loading = false; moreButton.disabled = false; } }
}
document.getElementById('history-search').addEventListener('submit', event => {
  event.preventDefault(); query = document.getElementById('history-query').value.trim(); const params = new URLSearchParams(); if (query) params.set('q', query); if (window.MemoryGardenWorkspace.id) params.set('workspace', window.MemoryGardenWorkspace.id);
  history.replaceState(null, '', '/history' + (params.size ? '?' + params : '')); moreButton.textContent = '查看更早的对话'; loadHistory(true);
});
moreButton.addEventListener('click', () => { moreButton.textContent = '查看更早的对话'; loadHistory(); });
query = new URLSearchParams(location.search).get('q') || ''; document.getElementById('history-query').value = query;
loadHistory(true);
