"""Pydantic models that drop unknown keywords, and a test helper that fails when they do (Chapter 33, Exercise 3).

``LooseParameters`` keeps Pydantic's default ``extra='ignore'``, like ToOp's ``PreprocessParameters``; a mistyped
keyword is accepted and does nothing. ``StrictParameters`` forbids extras, like ToOp's ``ActionSet``.
"""

from pydantic import BaseModel, ConfigDict, ValidationError

PROBE = 'no_such_field_42'


class LooseParameters(BaseModel):
    """Pydantic's default: unknown keywords are ignored."""

    preprocess_bb_outages: bool = False


class StrictParameters(BaseModel):
    """``extra='forbid'`` turns a typo into a ValidationError."""

    model_config = ConfigDict(extra='forbid')

    preprocess_bb_outages: bool = False


def extra_policy(model_cls):
    """What a model does with an unknown keyword: ``'forbid'`` (error), ``'ignore'`` (dropped) or ``'allow'`` (kept).

    The probe keyword is validated together with nothing else, so a model with required fields answers with
    "missing" errors; only an ``extra_forbidden`` error means the unknown keyword was rejected.
    """
    try:
        model = model_cls.model_validate({PROBE: 1})
    except ValidationError as err:
        return 'forbid' if any(e['type'] == 'extra_forbidden' for e in err.errors()) else 'ignore'
    return 'allow' if PROBE in model.model_dump() else 'ignore'


def assert_forbids_extras(model_cls):
    """The pytest helper: fails when ``model_cls`` accepts a keyword that is not one of its fields."""
    policy = extra_policy(model_cls)
    if policy != 'forbid':
        raise AssertionError(
            f'{model_cls.__name__} silently {"drops" if policy == "ignore" else "keeps"} unknown keywords'
        )


def main():
    """Show the silent drop and the two policies."""
    loose = LooseParameters(enable_bb_outage=True)
    print('LooseParameters(enable_bb_outage=True).model_dump() ->', loose.model_dump())
    for cls in (LooseParameters, StrictParameters):
        print(f'{cls.__name__}: {extra_policy(cls)}')
    try:
        StrictParameters(enable_bb_outage=True)
    except ValidationError as err:
        print('StrictParameters(enable_bb_outage=True) ->', err.errors()[0]['type'], err.errors()[0]['loc'])


if __name__ == '__main__':
    main()
