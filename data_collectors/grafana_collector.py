import json
import os
import sys

import numpy as np
from requests.auth import HTTPBasicAuth
import requests
import shutil
# import yaml

# with open('config.yaml', 'r') as file:
# # with open('config.yaml', 'r') as file:

#     config = yaml.safe_load(file)

def send_file_to_attachment(url_basic, auth_header, page_id, filename):
    """
    Функция для отправки файла в качестве вложения на страницу Confluence.

    :param url_basic: str, базовый URL для Confluence API
    :param auth_header: tuple, кортеж с данными для аутентификации (обычно логин и пароль или токен)
    :param page_id: str, идентификатор страницы Confluence, на которую нужно добавить вложение
    :param space: str, идентификатор пространства Confluence
    :param filename: str, путь к файлу, который нужно отправить в качестве вложения
    :return: response, ответ от сервера Confluence после попытки отправить файл
    """
    try:
        # Формирование URL для отправки вложения
        url = f"{url_basic}/rest/api/content/{page_id}/child/attachment"
        print(url)

        # Открытие файла
        with open(filename, 'rb') as file:
            files = {"file": file}

            # Отправка POST запроса с файлом и необходимыми заголовками
            response = requests.post(
                url,
                verify=False,
                files=files,
                auth=auth_header,
                headers=({'X-Atlassian-Token': 'nocheck'}),
            )

            # Возвращение ответа от сервера
            return response

    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")








def downloadImagesLogin(image_url, filename, username, password):
    """
    Функция для загрузки изображений по указанному URL и сохранения их в локальном каталоге.

    :param image_url: str, URL-адрес изображения, которое нужно загрузить
    :param filename: str, имя файла, под которым будет сохранено изображение
    :param username: str, логин для аутентификации
    :param password: str, пароль для аутентификации
    :return: str, имя сохраненного файла
    """

    try:
        # Отправка GET запроса для получения изображения с базовой аутентификацией
        r = requests.get(image_url, stream=True, auth=(username, password), verify=False)
        print(image_url)
        print(r.status_code)

        # Проверка статуса ответа
        if r.status_code == 200:
            r.raw.decode_content = True

            # Сохранение изображения в локальном каталоге
            with open(f'data_collectors/temporary_files/{filename}.jpg', 'wb') as f:
                shutil.copyfileobj(r.raw, f)

            print('Image sucessfully Downloaded: ', filename)
        else:
            print('Image Couldn\'t be retreived')

    except requests.exceptions.RequestException as e:
        print(f"Произошла ошибка при отправке запроса: {str(e)}")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

    return filename



def uploadFromGrafana(user, password, url_basic, space_conf, page_id, util_metrics, service, grafana_login, grafana_pass):
    """
    Функция для загрузки изображений из Grafana, отправки их на Confluence и удаления локальных копий.

    :param user: str, имя пользователя Grafana
    :param password: str, пароль пользователя Grafana
    :param url_basic: str, базовый URL для Confluence API
    :param space_conf: str, идентификатор пространства Confluence
    :param page_id: str, идентификатор страницы Confluence, на которую нужно добавить изображение
    :param util_metrics: list, список метрик в формате [(имя метрики, URL-адрес изображения), ...]
    :param service: str, имя сервиса
    :return: list, список изображений с разметкой для вставки на страницу Confluence
    """
    utils = ""

    for metric in util_metrics:
        try:
            # Загрузка изображений
            downloadImagesLogin(metric[1], metric[0] + '_' + service + '_' + page_id, grafana_login, grafana_pass)

            # Подготовка аргументов для отправки файла на Confluence
            auth = HTTPBasicAuth(user, password)
            file_path = f'data_collectors/temporary_files/{metric[0]}_{service}_{page_id}.jpg'

            # Отправка файла на Confluence
            send_file_to_attachment(url_basic, auth, page_id, file_path)

            # Формирование разметки для вставки изображения на страницу Confluence
            # utils.append(
            #     f'<ac:image><ri:attachment ri:filename="{metric[0]}_{service}_{page_id}.jpg" /></ac:image>'
            # )
            utils = f'<ac:image><ri:attachment ri:filename="{metric[0]}_{service}_{page_id}.jpg" /></ac:image>'

            # Удаление локальной копии файла
            os.remove(file_path)

        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

    return utils



def uploadFromGrafanaLogin(user, password, url_basic, space_conf, page_id, util_metrics, service):
    """
    Функция для загрузки изображений из Grafana, отправки их на Confluence и удаления локальных копий.

    :param user: str, имя пользователя Grafana
    :param password: str, пароль пользователя Grafana
    :param url_basic: str, базовый URL для Confluence API
    :param space_conf: str, идентификатор пространства Confluence
    :param page_id: str, идентификатор страницы Confluence, на которую нужно добавить изображение
    :param util_metrics: list, список метрик в формате [(имя метрики, URL-адрес изображения), ...]
    :param service: str, имя сервиса
    :return: list, список изображений с разметкой для вставки на страницу Confluence
    """
    utils = []

    for metric in util_metrics:
        try:
            # Загрузка изображений
            downloadImagesLogin(metric[1], metric[0] + '_' + service + '_' + page_id, user, password)

            # Подготовка аргументов для отправки файла на Confluence
            auth = HTTPBasicAuth(user, password)
            file_path = f'data_collectors/temporary_files/{metric[0]}_{service}_{page_id}.jpg'

            # Отправка файла на Confluence
            send_file_to_attachment(url_basic, auth, page_id, file_path)

            # Формирование разметки для вставки изображения на страницу Confluence
            utils.append([
                metric[0],
                f'<ac:image><ri:attachment ri:filename="{metric[0]}_{service}_{page_id}.jpg" /></ac:image>'
            ])

            # Удаление локальной копии файла
            # os.remove(file_path)

        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

    return utils



def getGroupStorage(utils):
    """
    Функция для формирования разметки хранилища Confluence с использованием вкладок.

    :param utils: list, список изображений с разметкой для вставки на страницу Confluence
    :param page_id: str, идентификатор страницы Confluence
    :param orientation: str, ориентация вкладок (не используется в текущей реализации)
    :return: str, разметка хранилища с вкладками для страницы Confluence
    """
    try:
        storage = "<ac:structured-macro ac:name=\"ui-tabs\"><ac:rich-text-body>"

        for util in utils:
            storage += (
                f"<ac:structured-macro ac:name=\"ui-tab\">"
                f"<ac:parameter ac:name=\"title\">{util[0]}</ac:parameter>"
                f"<ac:rich-text-body> <p>"
                f"{util[1]}"
                f"</p></ac:rich-text-body>"
                f"</ac:structured-macro>"
            )

        storage += "</ac:rich-text-body></ac:structured-macro>"

        print(storage)

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        storage = ""

    return storage



def getGroupGroupStorage(util, page_id, orientation, storages):
    """
    Функция для формирования разметки хранилища Confluence с использованием вкладок и группировки.

    :param util: list, список изображений с разметкой для вставки на страницу Confluence (не используется)
    :param page_id: str, идентификатор страницы Confluence
    :param orientation: str, ориентация вкладок (не используется в текущей реализации)
    :param storages: list, список хранилищ с названиями для группировки во вкладках
    :return: str, разметка группированных хранилищ с вкладками для страницы Confluence
    """
    try:
        storage = "<ac:structured-macro ac:name=\"ui-tabs\"><ac:rich-text-body>"

        for strg in storages:
            storage += (
                f"<ac:structured-macro ac:name=\"ui-tab\">"
                f"<ac:parameter ac:name=\"title\">{strg[1]}</ac:parameter>"
                f"<ac:rich-text-body> <p>"
                f"{strg[0]}"
                f"</p></ac:rich-text-body>"
                f"</ac:structured-macro>"
            )

        storage += "</ac:rich-text-body></ac:structured-macro>"

        print(storage)

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        storage = ""

    return storage


def getGroupDashboardStorage(dashboardTitle, utils, orientation):
    storage = "<h2>" + dashboardTitle + "</h2>"


    for index, util in enumerate(utils):
        if (index == 0) and (util[0] == "row"):
            storage = storage + "<h3>" + util[1] + "</h3><ac:structured-macro ac:name=\"ui-tabs\"><ac:rich-text-body>\
            <p class=\"auto-cursor-target\"><br/></p>"
            continue
        elif (index == 0):
            storage = storage + "<ac:structured-macro ac:name=\"ui-tabs\"><ac:rich-text-body>\
                        <p class=\"auto-cursor-target\"><br/></p><ac:structured-macro ac:name=\"ui-tab\">\
                    <ac:parameter ac:name=\"title\">" + util[1] + "</ac:parameter>\
                    <ac:rich-text-body><p>\
                    <ac:image ac:width=\"1100\">\
                            <ri:attachment ri:filename=\"" + util[0] + "\" />\
                    </ac:image>\
                    </p></ac:rich-text-body>\
                    </ac:structured-macro><p class=\"auto-cursor-target\"><br/></p>"
            continue
        elif (util[0] == "row"):
            storage = storage + "</ac:rich-text-body></ac:structured-macro>" + "<h3>" + util[1] + "</h3>\
            <ac:structured-macro ac:name=\"ui-tabs\"><ac:rich-text-body>\
            <p class=\"auto-cursor-target\"><br/></p>"
            continue
        else:
            storage = storage + "<ac:structured-macro ac:name=\"ui-tab\">\
                    <ac:parameter ac:name=\"title\">" + util[1] + "</ac:parameter>\
                    <ac:rich-text-body><p>\
                    <ac:image ac:width=\"1100\">\
                            <ri:attachment ri:filename=\"" + util[0] + "\" />\
                    </ac:image>\
                    </p></ac:rich-text-body>\
                    </ac:structured-macro><p class=\"auto-cursor-target\"><br/></p>"
    storage = storage + "</ac:rich-text-body></ac:structured-macro>"
    # print(storage)
    return storage



def getBISGroupStorage(utils):
    storage = "<ac:structured-macro ac:name=\"ui-tabs\"><ac:rich-text-body> \
            <p class=\"auto-cursor-target\"><br/></p>"
#<ac:image><ri:attachment ri:filename=\""+util[1]+"\" /></ac:image> \
    for util in utils:
        storage = storage + "<ac:structured-macro ac:name=\"ui-tab\"> \
                <ac:parameter ac:name=\"title\">" + util[0] + "</ac:parameter> \
                <ac:rich-text-body> \
                " + util[1] + "</ac:rich-text-body> \
                </ac:structured-macro>"
    storage = storage + "</ac:rich-text-body> \
         </ac:structured-macro>"

    return storage


def grafanaGraphicsUrl(host, uid, token, pathResult, timeStart, timeTo):
    graphicsUrl = []
    # Url to request for finding all accessible dashboards with basic auth
    print('\nStart to work with %s' % host)
    url = host + "/api/search"
    headers = {'Authorization': 'Bearer ' + token}
    r = requests.get(url, headers=headers, verify=False)

    # Проверка доступности страницы
    if (r.status_code == 200):
        print("Grafana is accessible! Congrats!")
        if (r.headers['content-type'] != "application/json"):
            sys.exit()
    else:
        print("Grafana is not available, status code: ", r.status_code)
        sys.exit()

    # Start reading url text content which contains multiple JSON objects
    jsonText = json.loads(r.text)
    if (len(jsonText) == 0):
        print("This Grafana has no dashboards. Please, check \"grafana.json\" file in the project folder")
        sys.exit()

    # Find title (uri) by uid
    print("Start searching for needed dashboard by ", uid)
    for item in jsonText:
        if (item['uid'] == uid):
            # print(item)
            uriParts = item['uri'].split("/")
            uri = uriParts[1]
            print("We have found the needed %s dashboard с именем %s" % (uid, item['title']))
            break

    # Create the url request to find all graphics in dashboard
    urlReqDB = host + "/api/dashboards/uid/" + uid
    print("URL of dashboard by uid is ", urlReqDB)
    reqDB = requests.get(urlReqDB, headers=headers, verify=False)
    # print("text of reqDB\n", reqDB.text)

    # Page content to JSON format
    allGraphics = json.loads(reqDB.text)

    dashboardTitle = allGraphics['dashboard']['title']
    # print("Dashboard title is ", dashboardTitle)

    # Get variables
    varsTitle = allGraphics['dashboard']['templating']['list']
    varsNumber = len(varsTitle)
    # print("Number of dashboard variables is ", varsNumber)
    varsName = []
    varsValue = []
    for item in varsTitle:
        varsName.append(item['name'])
        # Check if value is a string we should convert to array, so loop after can be work without separating it by a letter
        if (type(item['current']['text']) == type("str")):
            varsValue.append([item['current']['text']])
        else:
            varsValue.append(item['current']['text'])
    # print(varsName, varsValue)

    # Count how many panels dashboard has
    panelsNumber = allGraphics['dashboard']['panels']
    print("This dashboard has " + str(len(panelsNumber)) + " panels")

    # # Create the url request for Grafana Image Renderer
    urlReqGraph = host + "/render/d-solo/" + uid + "/" + uri + "?orgId=1&from=" + str(timeStart) + "&to=" + str(timeTo) + "&theme=light&width=1000&height=500&tz=Europe%2FMoscow&panelId="
    #
    # headers = [('Authorization', 'Bearer ' + token)]
    # opener = urllib.request.build_opener()
    # opener.addheaders = headers
    # urllib.request.install_opener(opener)

    # for panel in range(panelsNumber):
    for panel in panelsNumber:
        # print(panel)
        # print(panel['type'])
        if (panel['type'] == "graph"):
            panelId = panel['id']
            panelTitle = panel['title']

            # Проверяем если вместо заголовка переменная
            if "[[" in panelTitle:
                varTemp = panelTitle[2:-2]
                panelTitle = panel['scopedVars'][varTemp]['value']
            urlReqGraphNew = urlReqGraph + str(panelId)
            # print("URL before vars: " + urlReqGraphNew)
            if (len(varsTitle) > 0):
                for k in range(len(varsName)):
                    if (len(varsValue[k]) == 0):
                        break
                    else:
                        for j in range(len(varsValue[k])):
                            urlReqGraphNew = urlReqGraphNew + "&var-" + varsName[k] + "=" + varsValue[k][j]
                urlReqGraphNew = urlReqGraphNew.replace(" ", "%20").replace("[", "%5B").replace("]", "%5D")
            # print("Panel " + str(panelId) + ": " + urlReqGraphNew)
            fileName = "Image_" + uid + "_" + str(panelId)

            r = requests.get(urlReqGraphNew, stream=True, headers=headers, verify=False)
            if r.status_code == 200:
                r.raw.decode_content = True
                # urllib.request.urlretrieve(urlReqGraphNew, fileName)
                with open(pathResult + "\\" + fileName + '.jpg', 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                    f.close()
                # print('Image sucessfully Downloaded: ', fileName + '.jpg')
            else:
                print('Image Couldn\'t be retreived')
            graphicsUrl.append([fileName + '.jpg', panelTitle, urlReqGraphNew])
        if (panel['type'] == "row"):
            if ("[[" in panel['title']) and (panel['repeat'] == None) and ('repeatIteration' not in panel):
                continue
            if "[[" in panel['title']:
                varTempRow = panel['title'][2:-2]
                panelTitle = panel['scopedVars'][varTempRow]['value']
                graphicsUrl.append(['row', panelTitle, 'urlDoesNotNeed'])
            else:
                graphicsUrl.append(['row', panel['title'], 'urlDoesNotNeed'])
    # result = "<h2 class=\"auto-cursor-target\"><span style=\"color: rgb(0,51,102);\">     <strong>" + dashboardTitle +"</strong> </span> </h2> <p> <span style=\"color: rgb(0,51,102);\">*Цветом выделена ступень максимальной производительности</span> </p> <ac:structured-macro ac:macro-id=\"b39283d9-5567-4cfa-afb1-2d1068e9e957\" ac:name=\"ui-tabs\" ac:schema-version=\"1\"> <ac:rich-text-body> <p class=\"auto-cursor-target\"> <br/> </p> <ac:structured-macro ac:macro-id=\"2032033f-741a-4275-b0ab-7e58492b32c7\" ac:name=\"ui-tab\" ac:schema-version=\"1\"> <ac:parameter ac:name=\"title\">График выхода виртуальных пользователей</ac:parameter> <ac:rich-text-body> <p> <span style=\"color: rgb(0,51,102);\"> <em>Все пользователи были стартованы и выведены согласно расписания</em> </span> <span style=\"color: rgb(0,51,102);\"> <em> <br/> </em> </span> </p> <p> <ac:image> <ri:attachment ri:filename=\"image2022-12-30_12-19-28.png\"/> </ac:image> </p> </ac:rich-text-body> </ac:structured-macro> <p class=\"auto-cursor-target\"> <br/> </p> <ac:structured-macro ac:macro-id=\"b70a97c8-5e4e-431c-9ee5-e9598b3bef22\" ac:name=\"ui-tab\" ac:schema-version=\"1\"> <ac:parameter ac:name=\"title\">График усреднённого времени отправки сообщений MF LoadRunner</ac:parameter> <ac:rich-text-body> <p> <span style=\"color: rgb(0,51,102);\"> <em>Времена выполнения запросов по загрузке страниц сайта на протяжении всего теста соответствовали требованиям SLA.</em> </span> </p> <p> <ac:image> <ri:attachment ri:filename=\"image2022-12-30_12-24-10.png\"/> </ac:image> </p> </ac:rich-text-body> </ac:structured-macro> <p class=\"auto-cursor-target\"> <br/> </p> <ac:structured-macro ac:macro-id=\"f8e6695c-9f9a-4522-aed4-ea44c5b3ca9a\" ac:name=\"ui-tab\" ac:schema-version=\"1\"> <ac:parameter ac:name=\"title\">График распределения интенсивности отправки сообщений MF LoadRunner</ac:parameter> <ac:rich-text-body> <p> <span style=\"color: rgb(0,51,102);\"> <em> <ac:structured-macro ac:macro-id=\"e6a433e5-5dc5-4262-b82b-28a7ecb0615d\" ac:name=\"anchor\" ac:schema-version=\"1\"> <ac:parameter ac:name=\"\">Интенсивность</ac:parameter> </ac:structured-macro>Интенсивность запросов к сайту соответствует модели нагрузки.</em> </span> </p> <ac:image> <ri:attachment ri:filename=\"image2022-12-30_12-24-44.png\"/> </ac:image> </ac:rich-text-body> </ac:structured-macro> <p class=\"auto-cursor-target\"> <br/> </p> <ac:structured-macro ac:macro-id=\"0e89228c-a418-459f-ae72-bec544ba6afb\" ac:name=\"ui-tab\" ac:schema-version=\"1\"> <ac:parameter ac:name=\"title\">График распределения ошибок MF LoadRunner</ac:parameter> <ac:rich-text-body> <p> <span style=\"color: rgb(0,51,102);\"> <em>На графике можно видеть появление первой ошибки через 20 минут после после начала теста. </em> </span> <span style=\"color: rgb(0,51,102);\"> <em>Полный перечень ошибок см. <ac:link ac:anchor=\"Ошибки\"/> </em> </span> </p> <p> <ac:image> <ri:attachment ri:filename=\"image2022-12-30_12-25-24.png\"/> </ac:image> </p> </ac:rich-text-body> </ac:structured-macro> <p class=\"auto-cursor-target\"> <br/> </p> </ac:rich-text-body> </ac:structured-macro>"

    return dashboardTitle, graphicsUrl


def grafanaImageRenderer(project, timeStart, timeTo, pathResult):


    if project == "BIS":
        host = [config['grafana_bis']['grafana_host'], config['grafana_lre']['grafana_host']]
        uid = [config['grafana_bis']['uid'], config['grafana_lre']['uid']]
        token = [config['grafana_bis']['token'], config['grafana_lre']['token']]
    # elif project == "LRE":
    #     host = config['grafana_lre']['grafana_host']
    #     uid = config['grafana_lre']['uid']
    #     token = config['grafana_lre']['token']


    # host, uid, token = initGrafanaParam(project)

    storage = ""
    util_metrics_all = []
    for i in range(len(host)):
        dashboardTitle, util_metrics = grafanaGraphicsUrl(host[i], uid[i], token[i], pathResult, timeStart, timeTo)
        if len(util_metrics_all) == 0:
            util_metrics_all = util_metrics
        else:
            util_metrics_all = np.concatenate((util_metrics_all, util_metrics), axis=0)
        storage = storage + getGroupDashboardStorage(dashboardTitle, util_metrics, "False")
    return util_metrics_all, storage

