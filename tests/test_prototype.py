from pocket_genie.agent_loop import PrototypeAgentRunner
from pocket_genie.shared import ToolInvocation


def test_prototype_runner_executes_and_truncates():
    runner = PrototypeAgentRunner()
    runner.register_tool("ping", "Ping check", lambda x=1: {"ok": True, "x": x})
    reply = runner.execute_turn("check ping", ToolInvocation(id="c1", name="ping", arguments={"x": 5}))
    assert '"ok": true' in reply.content
