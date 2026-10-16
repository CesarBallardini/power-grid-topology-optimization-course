"""Which of ToOp's parameter and message models accept unknown keywords (Chapter 33, Exercise 3).

Run with ToOp's interpreter from this folder: ``../../../../../ToOp/.venv/Scripts/python.exe pydantic_extras_toop.py``.
It imports every module of ``toop_engine_interfaces.messages`` and of
``toop_engine_topology_optimizer.interfaces.messages``, collects the Pydantic models defined there and reports the
extra policy of each one with ``pydantic_extras.extra_policy``.
"""

import importlib
import inspect
import pkgutil

import pydantic_extras
from pydantic import BaseModel

PACKAGES = ('toop_engine_interfaces.messages', 'toop_engine_topology_optimizer.interfaces.messages')


def models_in(package_name):
    """Pydantic models defined (not merely imported) in the modules of a package, as ``{module: [class, ...]}``."""
    package = importlib.import_module(package_name)
    found = {}
    for info in pkgutil.walk_packages(package.__path__, package_name + '.'):
        if 'protobuf_schema' in info.name:
            continue
        module = importlib.import_module(info.name)
        classes = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, BaseModel) and cls is not BaseModel and cls.__module__ == module.__name__
        ]
        if classes:
            found[info.name.removeprefix(package_name + '.')] = classes
    return found


def policies(package_name):
    """``{(module, class name): policy}`` for every model of the package."""
    return {
        (module, cls.__name__): pydantic_extras.extra_policy(cls)
        for module, classes in models_in(package_name).items()
        for cls in classes
    }


def main():
    """Print the silent drop and the policy table of both packages."""
    from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
    from toop_engine_interfaces.stored_action_set import ActionSet

    params = PreprocessParameters(enable_bb_outage=True)
    dropped = 'enable_bb_outage' not in params.model_dump()
    print('PreprocessParameters(enable_bb_outage=True): keyword dropped from model_dump() ->', dropped)
    print('PreprocessParameters policy:', pydantic_extras.extra_policy(PreprocessParameters))
    print('ActionSet policy:', pydantic_extras.extra_policy(ActionSet))
    for package in PACKAGES:
        table = policies(package)
        counts = {policy: sum(1 for p in table.values() if p == policy) for policy in ('ignore', 'allow', 'forbid')}
        print(f'\n{package}: {len(table)} models, {counts}')
        for (module, name), policy in sorted(table.items()):
            print(f'  {policy:7s} {module}.{name}')


if __name__ == '__main__':
    main()
