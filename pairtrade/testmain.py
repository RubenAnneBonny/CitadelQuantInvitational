import numpy as np
import pandas as pd
import psycopg2

from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

critical_values = {0: {.9: 13.4294, .95: 15.4943, .99: 19.9349},
                   1: {.9: 2.7055, .95: 3.8415, .99: 6.6349}}

trace0_cv = critical_values[0][.95] # critical value for 0 cointegration relationships
trace1_cv = critical_values[1][.95] # critical value for 1 cointegration relationship

conn = psycopg2.connect(dbname="analysis", host="acer", user='yzhang2', password='analysis')
cursor = conn.cursor()

sqlstm = "SELECT * FROM yahooastock1d"
cursor.execute(sqlstm)
results = cursor.fetchall()
stock_data = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])
stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])

