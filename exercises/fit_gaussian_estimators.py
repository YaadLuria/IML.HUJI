import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

pio.templates.default = "simple_white"

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma, m = 10, 1, 1000
    ms = np.linspace(10, 1000, 100).astype(int)
    X = np.random.normal(mu, sigma, size=m)
    estimated_expectations = []

    ug = UnivariateGaussian()
    ug.fit(X)
    print('(', ug.mu_, ',', ug.var_, ')\n')

    # Question 2 - Empirically showing sample mean is consistent
    for m in ms:
        estimated_expectations.append(ug.fit(X[0:m]).mu_)

    go.Figure([go.Scatter(x=ms, y=estimated_expectations, fill=None,
                          mode="markers+lines",
                          line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=ms, y=[ug.mu_] * len(ms), fill='tonexty',
                          mode="markers+lines",
                          marker=dict(
                              color="black", size=1), showlegend=False)],
              layout=go.Layout(
                  title=r"$\text{(3.1.2) Plot of absolute distance "
                        r"between the estimated- and true value of the expectation}$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfX = ug.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdfX,
                          mode="markers",
                          line=dict(color="lightgrey"), showlegend=False)],
              layout=go.Layout(
                  title=r"$\text{(3.1.3) Plot of empirical PDF function "
                        r"under the fitted model}$",
                  xaxis_title="ordered sample values",
                  yaxis_title="PDFs",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    # from scipy.stats import multivariate_normal
    # y = multivariate_normal(mu, sigma, 1000)
    # y.f
    # print(y.cov)
    mg = MultivariateGaussian()
    X = np.random.multivariate_normal(mu, sigma, 1000)
    mg.fit(X)
    print(mg.mu_, end='\n')
    print(mg.cov_, end='\n')

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    log = np.zeros((200, 200))
    i = 0
    j = 0
    for ff1 in f1:
        for ff3 in f3:
            log[i, j] = mg.log_likelihood(np.array([ff1, 0, ff3, 0]), sigma,
                                          X)
            j += 1
        i += 1
        j = 0

    go.Figure(go.Heatmap(z=log, x=f1, y=f3),
              layout=go.Layout(
                  title=r"$\text{(3.2.4) Plot a heatmap of f 1 values as "
                        r"rows, f3 values as columns and the color being the calculated log likelihood"
                        r"under the fitted model}$",
                  xaxis_title="f1 values",
                  yaxis_title="f3 values",
                  height=300)).show()

    # Question 6 - Maximum likelihood
    print(np.max(log))
    print(np.unravel_index(log.argmax(), log.shape))
    print(np.where(log == np.max(log)))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
