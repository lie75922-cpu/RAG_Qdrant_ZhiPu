from ragforgex.core.registry import Registry


def test_registry_creates_components():
    registry = Registry()
    registry.register("value", lambda x: {"x": x})

    assert registry.create("value", x=3) == {"x": 3}
    assert registry.names() == ["value"]

