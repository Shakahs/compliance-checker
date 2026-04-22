import os


def greet(name: str) -> str:
    return f"hello {name}"


path = os.path.join("a", "b")
print(greet("world"))
