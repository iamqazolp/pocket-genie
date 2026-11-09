class SpecialistInfo {
  final String agentId;
  final String displayName;
  final String defaultTier;
  final List<String> domains;

  const SpecialistInfo({
    required this.agentId,
    required this.displayName,
    required this.defaultTier,
    required this.domains,
  });

  factory SpecialistInfo.fromJson(Map<String, dynamic> json) {
    return SpecialistInfo(
      agentId: json['agent_id'] as String? ?? '',
      displayName: json['display_name'] as String? ?? '',
      defaultTier: json['default_tier'] as String? ?? 't1',
      domains: (json['domains'] as List<dynamic>? ?? const [])
          .map((e) => e.toString())
          .toList(),
    );
  }
}

class QuantTelemetry {
  final String tier;
  final String profile;
  final double weightBits;
  final double effectiveActBits;
  final double kvMemorySavedMb;
  final bool prefixKvHit;

  const QuantTelemetry({
    required this.tier,
    required this.profile,
    required this.weightBits,
    required this.effectiveActBits,
    required this.kvMemorySavedMb,
    required this.prefixKvHit,
  });

  factory QuantTelemetry.fromJson(Map<String, dynamic> json) {
    return QuantTelemetry(
      tier: json['tier'] as String? ?? 't1',
      profile: json['profile'] as String? ?? 'P_fast',
      weightBits: (json['weight_bits'] as num?)?.toDouble() ?? 4.0,
      effectiveActBits: (json['effective_act_bits'] as num?)?.toDouble() ?? 8.0,
      kvMemorySavedMb:
          (json['kv_memory_saved_mb'] as num?)?.toDouble() ?? 0.0,
      prefixKvHit: json['prefix_kv_hit'] as bool? ?? true,
    );
  }
}

class TimelineStep {
  final String stepType;
  final String title;
  final String detail;
  final String specialist;
  final String tier;

  const TimelineStep({
    required this.stepType,
    required this.title,
    required this.detail,
    required this.specialist,
    required this.tier,
  });

  factory TimelineStep.fromJson(Map<String, dynamic> json) {
    return TimelineStep(
      stepType: json['step_type'] as String? ?? 'step',
      title: json['title'] as String? ?? '',
      detail: json['detail'] as String? ?? '',
      specialist: json['specialist'] as String? ?? 'supervisor',
      tier: json['tier'] as String? ?? 't1',
    );
  }
}

class ApprovalRequest {
  final String requestId;
  final String toolName;
  final Map<String, dynamic> arguments;
  final String reason;
  final String status;

  const ApprovalRequest({
    required this.requestId,
    required this.toolName,
    required this.arguments,
    required this.reason,
    required this.status,
  });

  factory ApprovalRequest.fromJson(Map<String, dynamic> json) {
    return ApprovalRequest(
      requestId: json['request_id'] as String? ?? '',
      toolName: json['tool_name'] as String? ?? '',
      arguments: (json['arguments'] as Map<String, dynamic>?) ?? const {},
      reason: json['reason'] as String? ?? '',
      status: json['status'] as String? ?? 'pending',
    );
  }
}

class GenieStatusResponse {
  final String sessionId;
  final String projectId;
  final String activeTier;
  final QuantTelemetry telemetry;
  final List<SpecialistInfo> specialists;
  final List<ApprovalRequest> pendingHitl;

  const GenieStatusResponse({
    required this.sessionId,
    required this.projectId,
    required this.activeTier,
    required this.telemetry,
    required this.specialists,
    required this.pendingHitl,
  });

  factory GenieStatusResponse.fromJson(Map<String, dynamic> json) {
    final specs = (json['specialists'] as List<dynamic>? ?? const [])
        .whereType<Map<String, dynamic>>()
        .map(SpecialistInfo.fromJson)
        .toList();
    final approvals = (json['pending_hitl'] as List<dynamic>? ?? const [])
        .whereType<Map<String, dynamic>>()
        .map(ApprovalRequest.fromJson)
        .toList();
    return GenieStatusResponse(
      sessionId: json['session_id'] as String? ?? '',
      projectId: json['project_id'] as String? ?? 'pocket-genie',
      activeTier: json['active_tier'] as String? ?? 't1',
      telemetry: QuantTelemetry.fromJson(
        (json['quant_telemetry'] as Map<String, dynamic>?) ?? const {},
      ),
      specialists: specs,
      pendingHitl: approvals,
    );
  }
}

class ChatTurnResponse {
  final String reply;
  final String rawReply;
  final String activeTier;
  final List<TimelineStep> steps;
  final ApprovalRequest? approvalRequest;
  final GenieStatusResponse state;

  const ChatTurnResponse({
    required this.reply,
    required this.rawReply,
    required this.activeTier,
    required this.steps,
    required this.approvalRequest,
    required this.state,
  });

  factory ChatTurnResponse.fromJson(Map<String, dynamic> json) {
    final stepsRaw = (json['steps'] as List<dynamic>? ?? const [])
        .whereType<Map<String, dynamic>>()
        .map(TimelineStep.fromJson)
        .toList();
    final approvalRaw = json['approval_request'];
    return ChatTurnResponse(
      reply: json['reply'] as String? ?? '',
      rawReply: json['raw_reply'] as String? ?? '',
      activeTier: json['active_tier'] as String? ?? 't1',
      steps: stepsRaw,
      approvalRequest: approvalRaw is Map<String, dynamic>
          ? ApprovalRequest.fromJson(approvalRaw)
          : null,
      state: GenieStatusResponse.fromJson(
        (json['state'] as Map<String, dynamic>?) ?? const {},
      ),
    );
  }
}
