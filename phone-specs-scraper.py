from bs4 import BeautifulSoup
import requests
import pandas as pd

import time

def get_full_specs_for_phone(html):
    
    specs_dict = {}

    soup = BeautifulSoup(requests.get(html).text, 'lxml')
    specs = soup.find('tbody').find_all('tr')
        
    try:
        specs_dict['Stars'] = soup.find('span', {'id': 'ave-stars'}).text
        specs_dict['Stars Count'] = soup.find('span', {'id': 'count-stars'}).text
        specs_dict['LikeShare'] = soup.find('div', {'class':'grid3 grid_2'}).a.text
    except:
        specs_dict['Stars'] = None
        specs_dict['Stars Count'] = None
        specs_dict['LikeShare'] = None

    for spec in specs:
        
        try:

            raw_specs = (
                spec
                .text
                .replace('\t', '')
                .replace('\r', '')
                .split('\n')
            )
            
            specs = list(filter(None, raw_specs))
        
            specs_dict[specs[0]] = specs[1]
        
        except:
            
            pass

    return specs_dict

def specs_list(
        base_html='https://www.pinoytechnoguide.com/smartphones/page/'
    ) -> str:
    
    try:

        page_num = 1
        last_page = False
        dne_text = 'does not exist'
        phone_spec_list = []
        
        while not last_page:
            # html should be https://www.pinoytechnoguide.com/smartphones/page/
            html = f'{base_html}{str(page_num)}'

            html_text = requests.get(html).text
            soup = BeautifulSoup(html_text, 'lxml')
            
            page_main_text = soup.find('div', {'id': 'main'}).text
            if dne_text in page_main_text:
                last_page = True
                break
                
            print(f'Current page: {page_num}')
            total_time = 0
            phones = soup.find_all('tr', {'class':'phone_block'})
            
            for phone in phones:
                
                t1 = time.time()
                
                phone_spec = {}
                name = phone.a.text.strip()
                link = phone.a['href']
                
                phone_spec['Name'] = name
                phone_spec['Link'] = link
                
                phone_spec.update(get_full_specs_for_phone(link)) 
                phone_spec_list.append(phone_spec)
                
                t2 = time.time()
                time_per_phone = t2 - t1
                print(f'{name:<60}: {time_per_phone:.3f}')
                total_time += time_per_phone
                
            print(f'Page {page_num} time: {total_time}')
            print('')
            page_num += 1
             
        print("---------DONE---------")
        
        return phone_spec_list
                  
    except Exception as e:
        
        print(e)
        return
    
def main():

    t1 = time.time()
    df = pd.DataFrame(specs_list())
    t2 = time.time()
    print(f'Over All time: {t2-t1}')
    df.to_csv('phone_specs_raw.csv', index=False)

if __name__ == '__main__':

    main()
