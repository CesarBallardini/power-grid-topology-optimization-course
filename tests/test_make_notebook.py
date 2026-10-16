import nbformat

from make_notebook import build, parse_percent

SCRIPT = """# %% [markdown]
# # Title
#
# A paragraph with $x^2$.
# %%
x = 2
print(x)
# %% [markdown]
# Second note.
"""


def test_parse_percent_splits_markdown_and_code():
    cells = parse_percent(SCRIPT)
    assert [k for k, _ in cells] == ['markdown', 'code', 'markdown']
    assert cells[0][1] == '# Title\n\nA paragraph with $x^2$.'
    assert cells[1][1] == 'x = 2\nprint(x)'


def test_build_is_a_valid_stripped_notebook():
    nb = build(parse_percent(SCRIPT))
    nbformat.validate(nb)
    assert nb.metadata['kernelspec']['name'] == 'python3'
    assert [c['id'] for c in nb.cells] == ['0', '1', '2']  # what nbstripout writes, so the hook is a no-op
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert len(code) == 1 and code[0].outputs == [] and code[0].execution_count is None
