import pydantic_extras
import pytest
from pydantic import BaseModel, ConfigDict


def test_loose_model_drops_the_keyword():
    assert pydantic_extras.LooseParameters(enable_bb_outage=True).model_dump() == {'preprocess_bb_outages': False}
    assert pydantic_extras.extra_policy(pydantic_extras.LooseParameters) == 'ignore'


def test_strict_model_forbids_the_keyword():
    assert pydantic_extras.extra_policy(pydantic_extras.StrictParameters) == 'forbid'
    pydantic_extras.assert_forbids_extras(pydantic_extras.StrictParameters)


def test_helper_fails_on_a_loose_model():
    with pytest.raises(AssertionError, match='silently drops'):
        pydantic_extras.assert_forbids_extras(pydantic_extras.LooseParameters)


def test_policy_with_required_fields_and_allow():
    class Required(BaseModel):
        name: str

    class Allowing(BaseModel):
        model_config = ConfigDict(extra='allow')

    assert pydantic_extras.extra_policy(Required) == 'ignore'
    assert pydantic_extras.extra_policy(Allowing) == 'allow'
