import ipaddress
import re
import requests
import bs4
import urllib.request
from datetime import date
from dateutil.parser import parse as date_parse

# Calculates number of months
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def get_id_having_address(url):
    try:
        ipaddress.ip_address(url)
        # print (ipaddress.ip_address(url))
        return 1
    except:
        return -1

def get_global_rank(domain):
    # return -1
    try:
        rank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {
            "name": domain
        })
    except:
        return -1

    # Extracts global rank of the website
    try:
        global_rank = int(re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0])
    except:
        global_rank = -1
    return global_rank

def get_https_domain(domain):  
    # HTTPS_token
    if re.findall("^https\-", domain):
        return -1
    else:
        return 1

def url_length_RT(url):
    if len(url) < 54:
        return 1
    elif len(url) >= 54 and len(url) <= 75:
        return 0
    else:
       return -1

def SubmitInfoToEmail(response):
    # Submitting_to_email
    # return 1

    if response == "":
        return -1
    if re.findall(r"[mail\(\)|mailto:?]", response.text):
        return 1
    else:
        return -1
def AbnormalFormAction(response):
    # Abnormal_URL
    if response.text == "":
        return 1
    else:
        return -1

def RightClickDisable(response):
    # RightClick
    if re.findall(r"event.button ?== ?2", response.text):
        return 1
    else:
        return -1

def PopUpWindow(response):
    # popUpWidnow
    if re.findall(r"alert\(", response.text):
        return 1
    else:
        return -1
def IframeOrFrame(response):
    # Iframe
    if re.findall(r"[<iframe>|<frameBorder>]", response.text):
        return 1
    else:
        return -1

def Preprocess(url):
    urlRE = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    try:
        html_page = urllib.request.urlopen(url).read()
    except Exception as e:
        return 0, 0, 0, 0
    # html_page = urllib.request.urlopen(url)
    soup = bs4.BeautifulSoup(html_page, features="html.parser")

    hyperlinkList = soup.findAll('a')

    TotalHyperlink = len(hyperlinkList)
    TotalExternal = 0
    TotalURL = 0
    curURL = url

    for link in hyperlinkList:
        destination = link.get('href')
        if destination != None:
            # return TotalHyperlink, TotalExternal, TotalURL, soup
        # print (destination)
            if destination.startswith(('http', 'ftp', 'www', '/')):
                TotalExternal += 1
            if re.match(urlRE, destination):
                TotalURL += 1
    return TotalHyperlink, TotalExternal, TotalURL, soup
def PctExtHyperlinks(TotalExternal, TotalHyperlink):
    # if curURL != url:
    #     TotalHyperlink, TotalExternal, TotalURL = Preprocess(url)
    if TotalHyperlink != 0:
        return (TotalExternal / TotalHyperlink)
    return 0
def PctExtResourceUrls(TotalURL, TotalHyperlink):
    # if curURL != url:
    #     Preprocess(url)
    if TotalHyperlink != 0:
        return (TotalURL / TotalHyperlink)
    return 0

def MissingTitle(url, soup):
    # if curURL != url:
    Preprocess(url)
    value = len(soup.findAll('title')) == 0
    if value == False:
        return 0
    return 1

def generate(url):
    # Converts the given URL into standard format
    if not re.match(r"^https?", url):
        url= "http://" + url
    # print (url)
    # Stores the response of the given URL
    try:
        response = requests.get(url)
    except:
        response = ""
    domain = re.findall(r"://([^/]+)/?", url)[0]
    features = []
    # if response == "":
    #     print (1)
    features.append(get_id_having_address(url))
    # print ("st")
    features.append(get_global_rank(domain))
    # print ("fs")
    features.append(get_https_domain(domain))
    features.append(get_https_domain(domain))
    # print (1)
    if response == "":
        features.append(-1)
        features.append(-1)
        features.append(-1)
        features.append(-1)
        features.append(-1)
    else :
        features.append(SubmitInfoToEmail(response))
        features.append(AbnormalFormAction(response))
        features.append(RightClickDisable(response))
        features.append(PopUpWindow(response))
        features.append(IframeOrFrame(response))
    features.append(url_length_RT(url))
    # print (2)
    TotalHyperlink, TotalExternal, TotalURL, soup = Preprocess(url)
    if TotalExternal == 0 and TotalURL == 0 and TotalHyperlink == 0:
        features.append(-1)
        features.append(-1)
        features.append(-1)
    else :
        features.append(PctExtHyperlinks(TotalExternal, TotalHyperlink))
        features.append(PctExtResourceUrls(TotalURL, TotalHyperlink))
        features.append(MissingTitle(url, soup))
    # print (features)
    return (features)
# generate("http://uaanow.com/admin/online/order.php?fav.1&amp;amp...")
# generate("http://hiroba.dqx.jp.isrel.usa.cc/account/app/svc/login.html")
