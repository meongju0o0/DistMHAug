CONFIG = {
    "cora": {
        'a_e': 10, 'b_e': 1, 'a_v': 5, 'b_v': 1,
        'kl': 0.5, 'h': 0.5,
        'sigma_delta_e': 0.1, 'sigma_delta_v': 0.01,
        'mu_e': 0.3, 'mu_v': 0.7,
        'lam1_e': 5, 'lam1_v': 0.9999, 'lam2_e': 1, 'lam2_v': 0.999
    },
    "citeseer": {
        'a_e': 5, 'b_e': 1, 'a_v': 10, 'b_v': 0.1,
        'kl': 0.2, 'h': 0.5,
        'sigma_delta_e': 0.005, 'sigma_delta_v': 0.1,
        'mu_e': 0.2, 'mu_v': 0.5,
        'lam1_e': 1, 'lam1_v': 0.9999, 'lam2_e': 10, 'lam2_v': 0.999
    },
    "ogb-product": {
        'a_e': 5, 'b_e': 1, 'a_v': 1, 'b_v': 1,
        'kl': 0.5, 'h': 0.5,
        'sigma_delta_e': 0.05, 'sigma_delta_v': 0.05,
        'mu_e': 0, 'mu_v': 0,
        'lam1_e': 1, 'lam1_v': 1, 'lam2_e': 1, 'lam2_v': 1
    },
}

# 'a_e', 'b_e', 'a_v', 'b_v': Target Distribution's variation
# 'kl', 'h': Loss Function
# 'sigma_delta_e', 'sigma_delta_v': Proposal Distribution's variation
# 'mu_e', 'mu_v': Mean of Change Ratio
# 'lam': Target Distribution Normalization
