# vk_text_classifier
Neural network classifier of vk (or even facebook) profiles based on texts corpora.

Usage:
#### Setting up config.py
```
VK_TOKEN = VK_TOKEN
FB_TOKEN = FB_TOKEN
```

#### Making corpora:
```
from util import CorporaClass
corpora_class = CorporaClass()

folder_name = "assets/corpora"
for filename in tqdm.tqdm(os.listdir(folder_name)):
    with open(f"{folder_name}/{filename}") as f:
        corpora_class.add_to_corpora(f)
```

#### Using API:
`python api.py`
```
>>> requests.post("http://0.0.0.0:9999/get_result", json={"name": "Georgiy", "user_vk": 134070307}).json()
{'name': 'Georgiy', 'results': [{'name': 'Искусство', 'value': 0.10368921607732773}, {'name': 'Политика', 'value': 0.14890874922275543}, {'name': 'Финансы', 'value': 0.15161116421222687}, {'name': 'Стратегическое управление', 'value': 0.25176137685775757}, {'name': 'Юриспруденция', 'value': 0.26189616322517395}, {'name': 'Исследования и разработки', 'value': 0.26951536536216736}, {'name': 'Промышленность', 'value': 0.15371596813201904}, {'name': 'Образование', 'value': 0.2749001979827881}, {'name': 'Благотворительность', 'value': 0.18278729915618896}, {'name': 'Здравоохранение', 'value': 0.11173231899738312}, {'name': 'Сельское хозяйство', 'value': 0.16613319516181946}, {'name': 'Государственное управление', 'value': 0.19677695631980896}, {'name': 'Реклама и маркетинг', 'value': 0.22656992077827454}, {'name': 'Инновации и модернизация', 'value': 0.26840078830718994}, {'name': 'Безопасность', 'value': 0.18020734190940857}, {'name': 'Военное дело', 'value': 0.28108736872673035}, {'name': 'Корпоративное управление', 'value': 0.2151733785867691}, {'name': 'Социальная защита', 'value': 0.2249830812215805}, {'name': 'Строительство', 'value': 0.15508094429969788}, {'name': 'Предпринимательство', 'value': 0.2644502818584442}, {'name': 'Спорт', 'value': 0.18960236012935638}, {'name': 'Инвестиции', 'value': 0.25759178400039673}]}
```
