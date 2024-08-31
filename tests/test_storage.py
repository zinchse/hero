from hero.storage import Storage, Node, Transition, Parameter, DISTANCE_THRESHOLD
import pytest

EPS = 1e-10


@pytest.fixture
def node_from() -> "Node":
    return Node(logical_plan="<plan_from1>", template_id=1, selectivities=[0.1], cardinalities=[1])


@pytest.fixture
def node_to() -> "Node":
    return Node(logical_plan="<plan_to1>", template_id=1, selectivities=[0.5], cardinalities=[2])


@pytest.fixture
def another_node_to() -> "Node":
    return Node(logical_plan="<plan_to2>", template_id=1, selectivities=[0.5], cardinalities=[2])


@pytest.fixture
def node_close_to_node_to() -> "Node":
    return Node(
        logical_plan="<plan_to1>",
        template_id=1,
        selectivities=[0.5 * (DISTANCE_THRESHOLD - EPS)],
        cardinalities=[2 * (DISTANCE_THRESHOLD - EPS)],
    )


@pytest.fixture
def unknown_node() -> "Node":
    return Node(logical_plan="<unknown_plan>", template_id=3, selectivities=[0.42], cardinalities=[42])


def test_default_scenario(node_from: "Node", node_to: "Node"):
    st = Storage()
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=0.5, parameter=Parameter(1, 666))
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=100, parameter=Parameter(2, 777))
    parameters = st.get_promised_parameters(node_from.template_id)
    assert parameters == {Parameter(1, 666), Parameter(2, 777)}


def test_collision_scenario(node_from: "Node", node_to: "Node", another_node_to: "Node"):
    st = Storage()
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=0.5, parameter=Parameter(1, 666))
    st.add_info(Transition(node_from=node_from, node_to=another_node_to), boost=100, parameter=Parameter(2, 777))
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=1000, parameter=Parameter(1, 6666))
    parameters = st.get_promised_parameters(node_from.template_id)
    assert parameters == {Parameter(1, 666), Parameter(1, 6666), Parameter(2, 777)}
    assert st.estimate_transition(Transition(node_from=node_from, node_to=node_to)) == (0.5 + 1000) / 2


def test_generalisation(node_from: "Node", node_to: "Node", node_close_to_node_to: "Node"):
    st = Storage()
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=0.5, parameter=Parameter(1, 666))
    assert st.estimate_transition(Transition(node_from=node_from, node_to=node_close_to_node_to)) == 0.5


def test_unknown_node_scenario(node_from: "Node", node_to: "Node", unknown_node: "Node"):
    st = Storage()
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=0.5, parameter=Parameter(1, 666))
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=100, parameter=Parameter(2, 777))
    st.add_info(Transition(node_from=node_from, node_to=node_to), boost=1000, parameter=Parameter(1, 6666))
    parameters = st.get_promised_parameters(unknown_node.template_id)
    assert parameters == set()
    assert st.estimate_transition(Transition(node_from=unknown_node, node_to=node_to)) == float("-inf")
    assert st.estimate_transition(Transition(node_from=unknown_node, node_to=unknown_node)) == 0.0
