from bs4 import BeautifulSoup
import pandas as pd
import requests


link = 'https://www.notebookcheck.net/Smartphone-Graphics-Cards-Benchmark-List.149363.0.html'
html = requests.get(link).text

soup = BeautifulSoup(html, 'lxml')

table = soup.find('table')
rows_odd = table.find_all('tr',  {'class':'smartphone_odd'})
rows_even = table.find_all('tr',  {'class':'smartphone_even'})
all_lists = rows_odd + rows_even

def gpu_score(row):

    name = row.find('td', {'class': 'specs fullname'}).text
    name_list = name.strip().split(maxsplit=1)
    if name_list[0] == 'PowerVR':
        name_used = name.strip()
    else:
        name_used = name_list[1]

    score = row.find('td', {'class': 'value bv_216'}).text
    score_used = score.strip().split('n')[0]
    if score_used == '':
        score_used = 0.0
    else:
        score_used = float(score_used)

    return name_used, score_used


gpu_data = {'gpu': [], 'gpu_score': []}
gpu_data['gpu'] = []
for row in all_lists:

    gpu_data['gpu'].append(gpu_score(row)[0])

    gpu_data['gpu_score'].append(gpu_score(row)[1])

gpu_score_df = pd.DataFrame(gpu_data)
gpu_score_df.to_csv('gpu_score.csv', index=False)