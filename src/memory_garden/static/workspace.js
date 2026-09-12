'use strict';
// Bind each loaded page to one library. Switching libraries starts a fresh page.
window.MemoryGardenWorkspace = (() => {
  let current = null, boundId = '', switching = false, dialog = null;
  const expectedId = new URLSearchParams(location.search).get('workspace');
  const el = (tag, cls, text) => { const item = document.createElement(tag); if (cls) item.className = cls; if (text !== undefined) item.textContent = text; return item; };
  const nameOf = item => item.public_demo_mode ? '演示笔记' : String(item.vault_path || '').split(/[\\/]/).filter(Boolean).pop() || '本地笔记库';
  function message(text, error = false) {
    document.querySelectorAll('[data-workspace-status]').forEach(item => { item.textContent = text; item.classList.toggle('error', error); });
    document.querySelectorAll('[data-workspace-home]').forEach(item => item.hidden = !error);
  }
  function renderCurrent() {
    document.querySelectorAll('[data-workspace-name]').forEach(item => item.textContent = nameOf(current));
    document.querySelectorAll('[data-workspace-path]').forEach(item => { if ('value' in item) item.value = current.vault_path || ''; else item.textContent = current.vault_path || ''; });
    message((current.counts?.sources ?? 0) + ' 篇已索引记录 · 原库只读');
  }
  const ready = fetch('/api/workspace').then(async response => {
    const data = await response.json();
    if (!response.ok || !data.workspace_id) throw new Error(data.error || '未能读取笔记库信息，请刷新页面。');
    current = data; boundId = expectedId || data.workspace_id; renderCurrent();
    if (data.restore_notice) message(data.restore_notice, true);
    if (boundId !== data.workspace_id) throw new Error('这个页面属于另一个笔记库。请从“回到当前笔记库”重新打开，避免读到同编号的其他对话。');
    return data;
  });
  ready.catch(error => message(error.message || '暂时连不上本地笔记库，请刷新后重试。', true));
  async function request(url, options = {}) {
    await ready;
    const response = await fetch(url, {...options, headers:{...options.headers, 'X-MG-Workspace-ID':boundId}});
    if (response.status === 409) {
      const data = await response.clone().json().catch(() => ({}));
      if (['workspace_changed','workspace_required'].includes(data.code)) message('笔记库已经切换。此页仍保留原来的内容，请回到当前笔记库后继续。', true);
    }
    return response;
  }
  function makeButton(label, action, cls = '') {
    const item = el('button', cls, label); item.type = 'button'; item.addEventListener('click', action); return item;
  }
  function createDialog() {
    dialog = el('dialog', 'vault-dialog'); dialog.setAttribute('aria-labelledby','vault-dialog-title');
    const top = el('div','vault-dialog-top');
    top.append(el('h2','','连接本地笔记库'), makeButton('关闭', () => { if (!switching) dialog.close(); }, 'text-button'));
    top.querySelector('h2').id = 'vault-dialog-title';
    const intro = el('p','quiet','输入 Obsidian 笔记库的根目录。原始笔记留在原处，每个库的对话和记忆分别保存。');
    const form = el('form','vault-connect-form'), label = el('label','','本地文件夹路径'), input = el('input');
    label.htmlFor = input.id = 'connect-vault-path'; input.name = 'vault_path'; input.required = true; input.autocomplete = 'off'; input.spellcheck = false; input.maxLength = 2000; input.placeholder = '例如 D:\\笔记\\我的花园';
    const submit = el('button','primary','连接并打开'); submit.type = 'submit';
    const status = el('p','vault-connect-status'); status.id = 'vault-connect-status'; status.setAttribute('role','status');
    const privacy = el('p','hint','首次连接只建立本地索引，不调用云端模型。需要模型交流时，再到这个库的设置中开启。');
    const recent = el('section','vault-recent'); recent.id = 'vault-recent';
    form.append(label,input,privacy,submit,status); form.addEventListener('submit', event => { event.preventDefault(); connect(input.value.trim()); });
    dialog.addEventListener('cancel', event => { if (switching) event.preventDefault(); });
    dialog.append(top,intro,form,recent); document.body.append(dialog);
  }
  async function connect(path) {
    if (switching || !path) return;
    switching = true;
    dialog.querySelectorAll('button,input').forEach(item => item.disabled = true);
    const status = document.getElementById('vault-connect-status'); status.classList.remove('error'); status.textContent = '正在读取本地笔记并准备关系图…';
    document.dispatchEvent(new CustomEvent('workspace:before-switch'));
    try {
      const response = await request('/api/workspace/connect', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({vault_path:path})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '连接没有完成，请检查文件夹路径。');
      if (!result.workspace_id) throw new Error('服务尚未确认新笔记库，请刷新后核对当前连接。');
      status.textContent = '已连接，正在打开你的关系图…';
      location.assign('/?' + new URLSearchParams({view:'map',scope:'global',workspace:result.workspace_id}));
    } catch (error) {
      status.textContent = error.message || '连接中断，请刷新页面核对当前笔记库后重试。'; status.classList.add('error');
      switching = false; dialog.querySelectorAll('button,input').forEach(item => item.disabled = false);
    }
  }
  async function open() {
    try {
      await ready;
      if (!dialog) createDialog();
      const response = await request('/api/workspace');
      if (!response.ok) throw new Error('暂时读不到笔记库列表，请刷新后重试。');
      const latest = await response.json();
      if (latest.workspace_id !== boundId) throw new Error('笔记库已经切换，请先回到当前笔记库。');
      const recent = document.getElementById('vault-recent'); recent.replaceChildren();
      const entries = latest.workspaces || [];
      if (entries.length) {
        recent.append(el('h3','','已连接的笔记库'));
        entries.forEach(item => {
          const row = el('div','vault-recent-row'), description = el('div');
          description.append(el('strong','',nameOf(item)),el('p','source-path',item.vault_path)); row.append(description);
          row.append(item.active ? el('span','quiet','当前使用') : makeButton('切换到 ' + nameOf(item), () => connect(item.vault_path))); recent.append(row);
        });
      }
      document.getElementById('vault-connect-status').textContent = '';
      dialog.showModal(); document.getElementById('connect-vault-path').focus();
    } catch (error) { message(error.message || '暂时无法打开连接窗口，请刷新后重试。', true); }
  }
  document.querySelectorAll('[data-connect-vault]').forEach(item => item.addEventListener('click', open));
  return {ready, fetch:request, get id(){return boundId;}, open};
})();
