import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df.loc[df['Temp'] < 60]
    df = df.loc[df['Temp'] > -60]
    df = df.loc[df['Day'] >= 1]
    df = df.loc[df['Day'] <= 31]
    return df


def question2(data: pd.DataFrame):
    dataIsrael = data[data['Country'] == 'Israel']
    years = dataIsrael['Year'].unique()
    fig = go.Figure()
    for year in years:
        thisYear = dataIsrael.loc[dataIsrael['Year'] == year]
        fig.add_trace(go.Scatter(x=thisYear['DayOfYear'], y=thisYear[
            'Temp'], mode="markers", name=str(year)))
    fig.layout = go.Layout(title="Temp for years in Israel",
                           xaxis={"title": "days"},
                           yaxis={"title": "temp"})
    fig.write_image("3.2.2 temp for days.png")

    groupMonth = dataIsrael[['Month', 'Temp']]
    groupMonth = groupMonth.groupby('Month').agg(np.std)
    px.bar(groupMonth, title="std of temps by months").write_image("3.2.2 "
                                                                   "temp std "
                                                                   "by "
                                                                   "month.png")


def question3(data: pd.DataFrame):
    countries = data['Country'].unique()
    fig = px.line(title="Countries Month and Temps")
    for country in countries:
        thisCountry = data.loc[data['Country']
                               == country]
        groupCountriesMonth = thisCountry.groupby('Month').agg(
            std=pd.NamedAgg(column='Temp', aggfunc=np.std),
            avg=pd.NamedAgg(column='Temp', aggfunc=np.average))

        tempFig = px.line(groupCountriesMonth, y='avg', error_y='std',
                          color=px.Constant(country))
        fig.add_traces(tempFig.data[0])

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("C:\CS\IML\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # raise NotImplementedError()
    # question2(data)

    # Question 3 - Exploring differences between countries
    question3(data)

    # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
