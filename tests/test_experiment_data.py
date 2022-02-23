
import numpy as np

from simulate_data.experiment_data import save_experiment_data, load_experiment_data, ExperimentData


m = 900
n1 = 12
n2 = 53
horder = 4
theta_dim = horder + 3

# Create dummy experiment data object.
data = ExperimentData(name="test",
                      snr=2000,
                      y=np.random.randn(m),
                      y_sd=np.ones(m),
                      f_true=np.random.randn(n1, n2),
                      f_ref=np.zeros((n1, n2)),
                      theta_true=np.random.randn(theta_dim),
                      theta_guess=np.random.randn(theta_dim),
                      theta_sd=np.ones(theta_dim),
                      hermite_order=horder
                      )
savename="test_data"

def test_save_experiment_data():
    # Save it
    save_experiment_data(data=data, savename=savename)


def test_load_experiment_data():
    loaded_data = load_experiment_data(savedir=savename)
    # Check that this is equal to dummy object on a sample basis.
    assert np.isclose(data.y, loaded_data.y).all()
    assert np.isclose(data.f_true, loaded_data.f_true).all()
    assert np.isclose(data.snr, loaded_data.snr)


