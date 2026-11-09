import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/genie_models.dart';

class GenieApiClient {
  String baseUrl;
  final http.Client _http;

  GenieApiClient({
    this.baseUrl = 'http://127.0.0.1:8090',
    http.Client? httpClient,
  }) : _http = httpClient ?? http.Client();

  Uri _uri(String path) {
    final trimmed = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    return Uri.parse('$trimmed$path');
  }

  Future<GenieStatusResponse> fetchStatus() async {
    final resp = await _http
        .get(_uri('/api/v1/status'))
        .timeout(const Duration(seconds: 10));
    if (resp.statusCode != 200) {
      throw Exception('Status failed (${resp.statusCode}): ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    return GenieStatusResponse.fromJson(data);
  }

  Future<ChatTurnResponse> sendMessage({
    required String message,
    String? targetSpecialist,
  }) async {
    final payload = <String, dynamic>{
      'message': message,
      'agent_type':
          targetSpecialist == null ? 'supervisor' : 'direct-specialist',
      if (targetSpecialist != null) 'target_specialist': targetSpecialist,
    };
    final resp = await _http
        .post(
          _uri('/api/v1/chat'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        )
        .timeout(const Duration(seconds: 90));
    if (resp.statusCode != 200) {
      throw Exception('Chat failed (${resp.statusCode}): ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    return ChatTurnResponse.fromJson(data);
  }

  Future<Map<String, dynamic>> resolveApproval({
    required String requestId,
    required bool approved,
  }) async {
    final encodedId = Uri.encodeComponent(requestId);
    final resp = await _http
        .post(
          _uri('/api/v1/approvals/$encodedId'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({'approved': approved}),
        )
        .timeout(const Duration(seconds: 30));
    if (resp.statusCode != 200) {
      throw Exception('Approval failed (${resp.statusCode}): ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<GenieStatusResponse> resetSession() async {
    final resp = await _http
        .post(
          _uri('/api/v1/session/reset'),
          headers: const {'Content-Type': 'application/json'},
          body: '{}',
        )
        .timeout(const Duration(seconds: 10));
    if (resp.statusCode != 200) {
      throw Exception('Reset failed (${resp.statusCode}): ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    return GenieStatusResponse.fromJson(data);
  }
}
