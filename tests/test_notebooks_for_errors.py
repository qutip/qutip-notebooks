import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def get_notebook_errors(path):
    """ Executes a notebook and returns occured errors """
    # read the notebook
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    # setup execution
    proc = ExecutePreprocessor(timeout=1000, kernel_name='python3')
    proc.allow_errors = True
    proc.preprocess(nb)

    # collect errors
    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)
    return errors


def test_piqs_superradiance():
    errors = get_notebook_errors('examples/piqs_superradiance.ipynb')
    assert not errors


def test_piqs_entropy_purity():
    errors = get_notebook_errors('examples/piqs-entropy_purity.ipynb')
    assert not errors


def test_measure_trajectories_cat_kerrs():
    errors = get_notebook_errors(
        'examples/measures-trajectories-cats-kerr.ipynb')
    assert not errors
