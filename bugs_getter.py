import requests as rq
import pandas as pd
import os
import shutil
import time

base_url = "https://bugzilla.mozilla.org/rest"

def create_dataset(dataset_path: str) -> None:

    def get_products_list() -> list:
        products = list()
        products.extend(rq.get(
            base_url + "/product_selectable"
        ).json()['ids'])
        products.extend(rq.get(
            base_url + "/product_enterable"
        ).json()['ids'])
        products.extend(rq.get(
            base_url + "/product_accessible"
        ).json()['ids'])
        return products

    def get_product_information(id: int) -> dict:
        for _ in range(5):
            response = rq.get(
                base_url + f"/product/{id}",
            )
            if response.status_code in [200, 202, 204]:
                break
        return response.json()['products'][0]

    def get_product_bugs(name: str) -> list:
        for _ in range(10):
            response =  rq.get(
                base_url + "/bug",
                {"product": name, "limit": 0}
            )
            if response.status_code in [200, 202, 204] and 'bugs' in response.json().keys():
                break
            else:
                time.sleep(60)
        if response.status_code in [500, 502, 504]:
            raise Exception(response.reason)
        elif 'bugs' not in response.json().keys():
            raise Exception(response.json())
        return response.json()['bugs']

    try:
        shutil.rmtree(dataset_path)
    except:
        print(f"Path {dataset_path} doesn't exist")
    os.mkdir(dataset_path)

    companies_list = get_products_list()

    products_collected = 0
    products_without_bugs = 0
    products_not_collected = 0

    for c in set(companies_list):
        try: 
            company_data = get_product_information(c)

            try:
                bugs = get_product_bugs(company_data['name'])

                if len(bugs) > 0:
                    products_collected += 1
                    df = pd.DataFrame(bugs)
                    df.to_csv(f'{dataset_path}/{company_data["id"]} - {str(company_data["name"]).replace(" ", "_")}.csv', index=False)
                    print(f"INFO: File {company_data['id']}_{str(company_data['name']).replace(' ', '_')}.csv create sucessfully")
                else:
                    products_without_bugs += 1
                    print(f"WARN: Product {company_data['id']}_{str(company_data['name']).replace(' ', '_')} doesn\'t have bug reports")
            except Exception as e:
                products_not_collected += 1
                print(f"ERROR: Couldn\'t fetch bugs of the product {company_data['id']}_{str(company_data['name']).replace(' ', '_')} - {e.args}")
        except Exception as e:
            products_not_collected += 1
            print(f"ERROR: Couldn\'t fetch product {c} information - {e.args}")

    print(f"""

Created files: {products_collected}
Products without bug reports: {products_without_bugs}
Request error count: {products_not_collected}
    """)
