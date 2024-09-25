from surprise import SVD
from surprise.model_selection import train_test_split

def create_svd_model(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
    return SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

def train_model(data, model, test_size=0.25, random_state=42):
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
    model.fit(trainset)
    return model, testset