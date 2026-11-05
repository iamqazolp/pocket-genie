/**
 * Standalone API Client SDK for Pocket-Genie (/api/v1/*).
 * Can be imported directly into React, Vue, or ported 1:1 to Flutter (Dio/HttpClient).
 */
class PocketGenieClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async getStatus() {
    const res = await fetch(`${this.baseUrl}/api/v1/status`);
    if (!res.ok) throw new Error(`Status error: ${res.status}`);
    return res.json();
  }

  async sendChat(message, options = {}) {
    const res = await fetch(`${this.baseUrl}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, ...options }),
    });
    if (!res.ok) throw new Error(`Chat error: ${res.status}`);
    return res.json();
  }

  async resolveApproval(requestId, approved) {
    const res = await fetch(`${this.baseUrl}/api/v1/approvals/${encodeURIComponent(requestId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved }),
    });
    if (!res.ok) throw new Error(`Approval error: ${res.status}`);
    return res.json();
  }

  async resetSession() {
    const res = await fetch(`${this.baseUrl}/api/v1/session/reset`, { method: "POST" });
    if (!res.ok) throw new Error(`Reset error: ${res.status}`);
    return res.json();
  }
}

class PocketGenieWebUI {
  constructor(client) {
    this.client = client;
  }

  async init() {
    const status = await this.client.getStatus();
    this.updateTelemetryUI(status);
  }

  toggleDrawer() {
    document.getElementById("drawer-backdrop").classList.toggle("hidden");
    document.getElementById("system-drawer").classList.toggle("hidden");
  }

  updateTelemetryUI(status) {
    const q = status.quant_telemetry || {};
    const tier = status.active_tier || "t1";
    const profile = q.profile || "P_fast";
    const bits = Number(q.effective_act_bits || 8.0).toFixed(2);
    const kvSaved = Number(q.kv_memory_saved_mb || 0.0).toFixed(2);
    const promo = ((q.promoted_token_ratio || 0) * 100).toFixed(1);

    document.getElementById("workspace-label").textContent = `Workspace: ${status.project_id || "local"}`;
    document.getElementById("tier-pill-text").textContent = `${profile} · ${bits}b · -${kvSaved}MB KV`;
    document.getElementById("d-profile").textContent = `${tier} · ${profile}`;
    document.getElementById("d-bits").textContent = `${bits} bits`;
    document.getElementById("d-promo").textContent = `${promo}%`;
    document.getElementById("d-kv").textContent = `${kvSaved} MB`;

    const specContainer = document.getElementById("d-specialists");
    specContainer.innerHTML = (status.specialists || [])
      .map(
        (s) =>
          `<div class="tag-card"><strong>${this._esc(s.display_name)}</strong> (${s.default_tier})<br/><span style="color:#94a3b8">Domains: ${(s.domains || []).join(", ")}</span></div>`
      )
      .join("");

    const skillContainer = document.getElementById("d-skills");
    skillContainer.innerHTML = (status.skills || [])
      .map(
        (sk) =>
          `<div class="tag-card"><strong>${this._esc(sk.name)}</strong> (${sk.recommended_tier})<br/><span style="color:#94a3b8">${this._esc(sk.description)}</span></div>`
      )
      .join("");
  }

  appendUserMessage(text) {
    document.getElementById("welcome-state").classList.add("hidden");
    const list = document.getElementById("messages-list");
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="msg-bubble">${this._esc(text)}</div>`;
    list.appendChild(el);
    this._scrollToBottom();
  }

  appendAssistantTurn(turn) {
    const list = document.getElementById("messages-list");
    const el = document.createElement("div");
    el.className = "msg assistant";

    const steps = turn.steps || [];
    const q = (turn.state && turn.state.quant_telemetry) || {};
    const stepSummaryLabel = `${steps.length} agent step${steps.length === 1 ? "" : "s"} · ${turn.active_tier || "t1"} ${q.profile || "P_fast"} (${Number(q.effective_act_bits || 8).toFixed(2)}b)`;

    const stepsHtml = steps.length
      ? `<details class="steps-details">
           <summary>${this._esc(stepSummaryLabel)}</summary>
           <div class="step-list">
             ${steps
               .map(
                 (s) => `<div class="step-item">
                   <div class="step-item-title">${this._esc(s.title)}</div>
                   ${s.detail ? `<div class="step-item-detail">${this._esc(s.detail)}</div>` : ""}
                 </div>`
               )
               .join("")}
           </div>
         </details>`
      : "";

    const approval = turn.approval_request;
    const approvalHtml =
      approval && approval.status === "pending"
        ? `<div class="approval-card" id="approval-${this._esc(approval.request_id)}">
             <div class="approval-title">Approval Required: ${this._esc(approval.tool_name)}</div>
             <div class="approval-args">${this._esc(JSON.stringify(approval.arguments, null, 2))}</div>
             <div class="approval-actions">
               <button class="btn-approve" onclick="window.ui.decideApproval('${this._esc(approval.request_id)}', true)">Approve &amp; Run</button>
               <button class="btn-deny" onclick="window.ui.decideApproval('${this._esc(approval.request_id)}', false)">Deny</button>
             </div>
           </div>`
        : "";

    el.innerHTML = `
      <div class="msg-bubble">
        <div class="msg-text">${this._esc(turn.reply)}</div>
        ${approvalHtml}
        ${stepsHtml}
      </div>
    `;
    list.appendChild(el);
    this._scrollToBottom();
  }

  async sendQuick(message, options = {}) {
    await this._dispatchMessage(message, options);
  }

  async handleSubmit(event) {
    event.preventDefault();
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    await this._dispatchMessage(text);
  }

  async _dispatchMessage(message, options = {}) {
    this.appendUserMessage(message);
    const btn = document.getElementById("send-btn");
    btn.disabled = true;
    try {
      const turn = await this.client.sendChat(message, options);
      this.appendAssistantTurn(turn);
      if (turn.state) this.updateTelemetryUI(turn.state);
    } catch (err) {
      this.appendAssistantTurn({ reply: `Error: ${err.message}`, steps: [] });
    } finally {
      btn.disabled = false;
    }
  }

  async decideApproval(requestId, approved) {
    const res = await this.client.resolveApproval(requestId, approved);
    const card = document.getElementById(`approval-${requestId}`);
    if (card) {
      card.innerHTML = `<div class="approval-title" style="color:${approved ? "#34d399" : "#f87171"}">
        ${approved ? "Approved & Executed" : "Denied"}: ${this._esc(res.output)}
      </div>`;
    }
    if (res.state) this.updateTelemetryUI(res.state);
  }

  async resetChat() {
    const state = await this.client.resetSession();
    document.getElementById("messages-list").innerHTML = "";
    document.getElementById("welcome-state").classList.remove("hidden");
    this.updateTelemetryUI(state);
  }

  _scrollToBottom() {
    const container = document.getElementById("chat-container");
    container.scrollTop = container.scrollHeight;
  }

  _esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
}

window.PocketGenieClient = PocketGenieClient;
window.ui = new PocketGenieWebUI(new PocketGenieClient());
window.addEventListener("DOMContentLoaded", () => window.ui.init());
