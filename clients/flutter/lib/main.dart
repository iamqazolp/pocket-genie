import 'package:flutter/material.dart';
import 'api/genie_client.dart';
import 'models/genie_models.dart';

void main() {
  runApp(const PocketGenieApp());
}

class PocketGenieApp extends StatelessWidget {
  const PocketGenieApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pocket Genie',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0B0F17),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF38BDF8),
          secondary: Color(0xFF34D399),
          surface: Color(0xFF151C28),
        ),
        useMaterial3: true,
      ),
      home: const GenieChatScreen(),
    );
  }
}

class ChatBubbleItem {
  final String role;
  final String text;
  final String tier;
  final List<TimelineStep> steps;
  ApprovalRequest? approval;

  ChatBubbleItem({
    required this.role,
    required this.text,
    this.tier = 't1',
    this.steps = const [],
    this.approval,
  });
}

class GenieChatScreen extends StatefulWidget {
  const GenieChatScreen({super.key});

  @override
  State<GenieChatScreen> createState() => _GenieChatScreenState();
}

const String _kDefaultGenieApiUrl = String.fromEnvironment(
  'GENIE_API_URL',
  defaultValue: 'http://127.0.0.1:8090',
);

class _GenieChatScreenState extends State<GenieChatScreen> {
  final GenieApiClient _api = GenieApiClient(
    baseUrl: _kDefaultGenieApiUrl,
  );
  final TextEditingController _inputCtrl = TextEditingController();
  final ScrollController _scrollCtrl = ScrollController();

  bool _connected = false;
  bool _busy = false;
  String _activeTier = 't1';
  String _profile = 'P_fast';
  double _actBits = 8.0;
  String? _selectedSpecialist;
  List<SpecialistInfo> _specialists = const [];
  final List<ChatBubbleItem> _messages = [];

  @override
  void initState() {
    super.initState();
    _refreshStatus();
  }

  Future<void> _refreshStatus() async {
    try {
      final st = await _api.fetchStatus();
      if (!mounted) return;
      setState(() {
        _connected = true;
        _activeTier = st.activeTier;
        _profile = st.telemetry.profile;
        _actBits = st.telemetry.effectiveActBits;
        _specialists = st.specialists;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _connected = false);
    }
  }

  Future<void> _sendPrompt([String? preset]) async {
    final text = (preset ?? _inputCtrl.text).trim();
    if (text.isEmpty || _busy) return;
    if (preset == null) _inputCtrl.clear();

    setState(() {
      _busy = true;
      _messages.add(ChatBubbleItem(role: 'user', text: text));
    });
    _scrollToBottom();

    try {
      final res = await _api.sendMessage(
        message: text,
        targetSpecialist: _selectedSpecialist,
      );
      if (!mounted) return;
      setState(() {
        _connected = true;
        _activeTier = res.activeTier;
        _profile = res.state.telemetry.profile;
        _actBits = res.state.telemetry.effectiveActBits;
        _messages.add(
          ChatBubbleItem(
            role: 'assistant',
            text: res.reply,
            tier: res.activeTier,
            steps: res.steps,
            approval: res.approvalRequest,
          ),
        );
      });
    } catch (err) {
      if (!mounted) return;
      setState(() {
        _connected = false;
        _messages.add(
          ChatBubbleItem(
            role: 'assistant',
            text: 'Connection error: $err',
          ),
        );
      });
    } finally {
      if (mounted) {
        setState(() => _busy = false);
        _scrollToBottom();
      }
    }
  }

  Future<void> _handleApproval(
    ChatBubbleItem bubble,
    ApprovalRequest req,
    bool approved,
  ) async {
    setState(() => _busy = true);
    try {
      final res = await _api.resolveApproval(
        requestId: req.requestId,
        approved: approved,
      );
      if (!mounted) return;
      setState(() {
        bubble.approval = ApprovalRequest(
          requestId: req.requestId,
          toolName: req.toolName,
          arguments: req.arguments,
          reason: req.reason,
          status: approved ? 'approved' : 'rejected',
        );
        _messages.add(
          ChatBubbleItem(
            role: 'assistant',
            text: res['output']?.toString() ?? 'Action resolved.',
            tier: _activeTier,
          ),
        );
      });
    } catch (err) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Approval failed: $err')),
      );
    } finally {
      if (mounted) {
        setState(() => _busy = false);
        _scrollToBottom();
      }
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 220),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _openServerSettings() async {
    final ctrl = TextEditingController(text: _api.baseUrl);
    final updated = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF151C28),
      builder: (ctx) {
        return Padding(
          padding: EdgeInsets.only(
            left: 16,
            right: 16,
            top: 16,
            bottom: MediaQuery.of(ctx).viewInsets.bottom + 20,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Pocket Genie API Endpoint',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: ctrl,
                decoration: const InputDecoration(
                  labelText: 'Server URL',
                  hintText: 'http://127.0.0.1:8090',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 10),
              Wrap(
                spacing: 8,
                children: [
                  ActionChip(
                    label: const Text('USB / Termux (127.0.0.1)'),
                    onPressed: () => ctrl.text = 'http://127.0.0.1:8090',
                  ),
                  ActionChip(
                    label: const Text('Mac LAN (192.168.0.101)'),
                    onPressed: () => ctrl.text = 'http://192.168.0.101:8090',
                  ),
                ],
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.pop(ctx, ctrl.text.trim()),
                  child: const Text('Save & Reconnect'),
                ),
              ),
            ],
          ),
        );
      },
    );

    if (updated != null && updated.isNotEmpty) {
      setState(() => _api.baseUrl = updated);
      await _refreshStatus();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF111827),
        titleSpacing: 12,
        title: Row(
          children: [
            Icon(
              Icons.circle,
              size: 10,
              color: _connected
                  ? const Color(0xFF34D399)
                  : const Color(0xFFF87171),
            ),
            const SizedBox(width: 8),
            const Text(
              'Pocket Genie',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
            ),
            const SizedBox(width: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: const Color(0xFF1E293B),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                '$_activeTier · $_profile (${_actBits.toStringAsFixed(1)}b)',
                style: const TextStyle(fontSize: 11, color: Color(0xFF38BDF8)),
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Reset Session',
            icon: const Icon(Icons.refresh, size: 20),
            onPressed: () async {
              await _api.resetSession();
              setState(() => _messages.clear());
              await _refreshStatus();
            },
          ),
          IconButton(
            tooltip: 'Server Settings',
            icon: const Icon(Icons.settings_ethernet, size: 20),
            onPressed: _openServerSettings,
          ),
        ],
      ),
      body: Column(
        children: [
          _buildSpecialistBar(),
          Expanded(
            child: _messages.isEmpty
                ? _buildEmptyState()
                : ListView.builder(
                    controller: _scrollCtrl,
                    padding: const EdgeInsets.all(12),
                    itemCount: _messages.length,
                    itemBuilder: (_, idx) => _buildBubble(_messages[idx]),
                  ),
          ),
          if (_busy)
            const LinearProgressIndicator(minHeight: 2),
          _buildComposer(),
        ],
      ),
    );
  }

  Widget _buildSpecialistBar() {
    return Container(
      height: 44,
      color: const Color(0xFF111827),
      child: ListView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        children: [
          ChoiceChip(
            label: const Text('Auto (Supervisor)'),
            selected: _selectedSpecialist == null,
            onSelected: (_) => setState(() => _selectedSpecialist = null),
          ),
          const SizedBox(width: 6),
          for (final spec in _specialists) ...[
            ChoiceChip(
              label: Text(spec.displayName.split(' ').first),
              selected: _selectedSpecialist == spec.agentId,
              onSelected: (_) =>
                  setState(() => _selectedSpecialist = spec.agentId),
            ),
            const SizedBox(width: 6),
          ],
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    const prompts = [
      'Check git status of this repo',
      'Compare W4A8 vs W4A16 accuracy from workspace SQL',
      'Check Xiaomi 14 Pro battery and CPU health',
      'Search knowledge base for SmoothQuant outlier boundaries',
    ];
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.memory, size: 44, color: Color(0xFF38BDF8)),
            const SizedBox(height: 10),
            Text(
              'Connected to ${_api.baseUrl}',
              style: const TextStyle(fontSize: 13, color: Colors.white70),
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: [
                for (final p in prompts)
                  ActionChip(
                    label: Text(p, style: const TextStyle(fontSize: 12)),
                    onPressed: () => _sendPrompt(p),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBubble(ChatBubbleItem item) {
    final isUser = item.role == 'user';
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        constraints: const BoxConstraints(maxWidth: 340),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? const Color(0xFF0284C7) : const Color(0xFF151C28),
          borderRadius: BorderRadius.circular(14),
          border: isUser
              ? null
              : Border.all(color: const Color(0xFF1E293B), width: 1),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SelectableText(
              item.text,
              style: const TextStyle(fontSize: 14, height: 1.4),
            ),
            if (item.steps.isNotEmpty) ...[
              const SizedBox(height: 8),
              ExpansionTile(
                tilePadding: EdgeInsets.zero,
                dense: true,
                title: Text(
                  '${item.steps.length} execution step(s) · ${item.tier}',
                  style: const TextStyle(
                    fontSize: 12,
                    color: Color(0xFF94A3B8),
                  ),
                ),
                children: [
                  for (final s in item.steps)
                    ListTile(
                      dense: true,
                      contentPadding: EdgeInsets.zero,
                      title: Text(
                        s.title,
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      subtitle: s.detail.isNotEmpty
                          ? Text(
                              s.detail,
                              style: const TextStyle(
                                fontSize: 11,
                                fontFamily: 'monospace',
                              ),
                            )
                          : null,
                    ),
                ],
              ),
            ],
            if (item.approval != null) ...[
              const Divider(height: 16),
              Text(
                'Approval required: ${item.approval!.toolName}',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFFFBBF24),
                ),
              ),
              const SizedBox(height: 6),
              if (item.approval!.status == 'pending')
                Row(
                  children: [
                    FilledButton.tonal(
                      onPressed: _busy
                          ? null
                          : () => _handleApproval(item, item.approval!, true),
                      child: const Text('Approve'),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton(
                      onPressed: _busy
                          ? null
                          : () => _handleApproval(item, item.approval!, false),
                      child: const Text('Reject'),
                    ),
                  ],
                )
              else
                Text(
                  'Status: ${item.approval!.status.toUpperCase()}',
                  style: const TextStyle(fontSize: 11, color: Colors.white60),
                ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildComposer() {
    return SafeArea(
      top: false,
      child: Container(
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 10),
        color: const Color(0xFF111827),
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _inputCtrl,
                onSubmitted: (_) => _sendPrompt(),
                decoration: InputDecoration(
                  hintText: 'Ask Pocket Genie...',
                  isDense: true,
                  filled: true,
                  fillColor: const Color(0xFF1E293B),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(20),
                    borderSide: BorderSide.none,
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 10,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            IconButton.filled(
              onPressed: _busy ? null : _sendPrompt,
              icon: const Icon(Icons.send_rounded, size: 18),
            ),
          ],
        ),
      ),
    );
  }
}
