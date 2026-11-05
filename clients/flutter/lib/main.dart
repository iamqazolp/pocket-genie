import 'package:flutter/material.dart';

void main() {
  runApp(const PocketGenieBootstrapApp());
}

class PocketGenieBootstrapApp extends StatelessWidget {
  const PocketGenieBootstrapApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pocket Genie Mobile',
      theme: ThemeData.dark(useMaterial3: true),
      home: const Scaffold(
        body: Center(child: Text('Pocket Genie /api/v1 bootstrap')),
      ),
    );
  }
}
