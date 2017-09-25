import flask
import json
from redis import Redis

import jinja2
import numpy as np
from util import ResultClass


MESSAGE_INVALID_FIELDS = jinja2.Template(
    '{{ \', \'.join(fields)}} {% if fields|length>1 %}are{% else %}is{% endif %} invalid'
)
MESSAGE_IS_NOT_CORRECT = jinja2.Template('Error \'{{field}}\' is appeared')

app = flask.Flask(__name__)

redis_obj = Redis(host='redis', port=6379)
result = ResultClass(redis_obj)
labels = ["Искусство", "Политика", "Финансы", "Стратегическое управление", "Юриспруденция", "Исследования и разработки",
          "Промышленность", "Образование", "Благотворительность", "Здравоохранение", "Сельское хозяйство",
          "Государственное управление", "Реклама и маркетинг", "Инновации и модернизация", "Безопасность",
          "Военное дело", "Корпоративное управление", "Социальная защита", "Строительство", "Предпринимательство",
          "Спорт", "Инвестиции"]
margins = json.load(open("assets/margins.json"))


@app.route("/")
def index():
    return "Welcome", 200


@app.route("/get_result", methods=["POST"])
def get_result():
    """
    POST http://0.0.0.0:9999/get_result
    json={"name": NAME, "user_vk": VK_ID, "user_fb": FB_PAGE}
    :return:
    """
    data = flask.request.get_json()
    name = data.get('name')
    user_vk = data.get('user_vk')
    user_fb = data.get('user_fb')
    verbose = data.get("verbose", False)

    if user_vk is None and user_fb is None:
        return app.response_class(
            response=json.dumps({'status': 'error', 'message': "You must provide at least user_vk or user_fb"}),
            status=400,
            mimetype='application/json')

    try:
        verdict = result.get_result(user_vk, user_fb)
        print(f"Got verdict.")
        norm_names = dict(zip([a[0] for a in verdict], labels))

        interests = []
        accepted_cols = []
        for col, value in [(t[0], t[1]) for t in sorted(verdict, key=lambda x: x[1], reverse=True)][:11]:
            if verbose:
                interests.append({"name": norm_names[col], "value": float(value)})
            elif value > 0.85 * margins[col]:  # delim
                accepted_cols.append(col)
        if not verbose:
            result_cols = list(np.array(accepted_cols)[np.array([t[1] for t in verdict if t[0] in
                                                                 accepted_cols]).argsort()[::-1]][:5])
            interests = [norm_names[col] for col in result_cols]
        result.texts = []
        print(f"Interests: *{interests}")
        return app.response_class(
            response=json.dumps({"name": name, "interests": interests}),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return app.response_class(
            response=json.dumps({'status': 'error', 'message': MESSAGE_IS_NOT_CORRECT.render(field=e.args)}),
            status=400,
            mimetype='application/json')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
