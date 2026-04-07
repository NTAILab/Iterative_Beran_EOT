from sksurv.datasets import load_veterans_lung_cancer
from sklearn.model_selection import train_test_split
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

from save_results import save_results
from beran_baseline import beran_baseline
from beran_iterative_EOT import beran_iterative_EOT

if __name__ == '__main__':
    tau = 2.0
    test_size = 0.2
    random_seed = 123
    k = 3
    epochs_count = 10
    reg_pi = 0.05

    X, y = load_veterans_lung_cancer()
    time_column = 'Survival_in_days'
    cens_column = 'Status'
    X_encoded = OneHotEncoder().fit_transform(X)
    X = X_encoded.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    deltas_train = y_train[cens_column]
    deltas_test = y_test[cens_column]

    times_train = y_train[time_column].astype(np.float64)
    times_test = y_test[time_column].astype(np.float64)

    beran_b = beran_baseline(tau=tau)
    beran_b.fit(X_train_scaled, deltas_train, times_train)
    times, SF_baseline = beran_b.survival_curve(X_test_scaled)
    c_index, ibs = beran_b.calculate_metics(deltas_test, times_test, SF_baseline)
    metrics_baseline = {"c_index": c_index, "IBS": ibs}

    beran_new = beran_iterative_EOT(tau=tau, k=k, reg_pi=reg_pi)
    beran_new.fit(X_train_scaled, deltas_train, times_train)
    train_history = beran_new.train(X_test_scaled, deltas_test, times_test, epochs = epochs_count)

    tests = range(5)
    save_results(beran_b, beran_new, metrics_baseline, train_history, X_test_scaled[tests], deltas_test[tests], times_test[tests])
