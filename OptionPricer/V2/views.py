from django.shortcuts import render_to_response

import project, premium_project, premium_project_oop
import pdb

# Create your views here.
from django.http import HttpResponse


def option_pricer(request):
    return render_to_response('index.html')


def bs_euro(request):
    spot_price = request.GET.get("spot_price")
    volatility = request.GET.get("volatility")
    rate = request.GET.get("rate")
    maturity = request.GET.get("maturity")
    strike_price = request.GET.get("strike_price")
    option_type = request.GET.get("option_type")
    price = project.bs(spot_price, strike_price, maturity, volatility, rate, option_type)
    return HttpResponse(price)


def get_geometric_asian_option(request):
    # pdb.set_trace()
    spot_price = float(request.GET.get("spot_price"))
    volatility = float(request.GET.get("volatility"))
    rate = float(request.GET.get("rate"))
    maturity = float(request.GET.get("maturity"))
    strike_price = float(request.GET.get("strike_price"))
    option_type = request.GET.get("option_type")
    observation_num = float(request.GET.get("observation_num"))
    #     price=project.geometric_asian_option(K, T, R, V, S0, N, option_type)

    price = project.geometric_asian_option(strike_price, maturity, rate, volatility, spot_price, observation_num,
                                           option_type)
    return HttpResponse(price)


def get_geometric_basket_option(request):
    S1 = float(request.GET.get("S1"))
    S2 = float(request.GET.get("S2"))
    V1 = float(request.GET.get("V1"))
    V2 = float(request.GET.get("V2"))
    R = float(request.GET.get("R"))
    T = float(request.GET.get("T"))
    K = float(request.GET.get("K"))
    rou = float(request.GET.get("rou"))
    option_type = request.GET.get("option_type")
    price = project.geometric_basket_option(S1, S2, V1, V2, R, T, K, rou, option_type)
    return HttpResponse(price)


def get_arithmetic_asian_option(request):
    S0 = float(request.GET.get("spot_price"))
    V = float(request.GET.get("volatility"))
    R = float(request.GET.get("rate"))
    T = float(request.GET.get("maturity"))
    K = float(request.GET.get("strike_price"))
    geo_K = K
    option_type = request.GET.get("option_type")
    N = float(request.GET.get("observation_num"))
    path_num = int(request.GET.get("path_num"))
    control_variate = request.GET.get("control_variate")
    price, std, confcv = project.arithmetic_asian_option(K, geo_K, T, R, V, S0, N, option_type, path_num,
                                                         control_variate)
    return HttpResponse(str(price) + ':' + str(confcv))


def get_arithmetic_basket_option(request):
    S1 = float(request.GET.get("S1"))
    S2 = float(request.GET.get("S2"))
    V1 = float(request.GET.get("V1"))
    V2 = float(request.GET.get("V2"))
    R = float(request.GET.get("R"))
    T = float(request.GET.get("T"))
    K = float(request.GET.get("K"))
    geo_K = K
    rou = float(request.GET.get("rou"))
    option_type = request.GET.get("option_type")
    path_num = int(request.GET.get("path_num"))
    control_variate = request.GET.get("control_variate")
    price, std, confcv = project.arithmetic_basket_option(S1, S2, V1, V2, R, T, K, geo_K, rou, option_type, path_num,
                                                          control_variate)
    return HttpResponse(str(price) + ':' + str(confcv))


def goto_premium(request):
    return render_to_response('premium.html')


def get_standardMC_european_option(request):
    spot_price = float(request.GET.get("spot_price"))
    volatility = float(request.GET.get("volatility"))
    rate = float(request.GET.get("rate"))
    maturity = float(request.GET.get("maturity"))
    strike_price = float(request.GET.get("strike_price"))
    option_type = float(request.GET.get("option_type"))
    N = float(request.GET.get("observation_num"))
    path_num = int(request.GET.get("path_num"))
    # price, std, confcv = premium_project.standardMC_european_option(strike_price, maturity, rate, volatility,
    #                                                                 spot_price, N, option_type, path_num)
    price, std, confcv = premium_project_oop.standardMC_european_option(strike_price, maturity, rate, volatility,
                                                                    spot_price, N, option_type, path_num)
    return HttpResponse(str(price) + ':' + str(confcv))


def get_GPU_european_option(request):
    spot_price = float(request.GET.get("spot_price"))
    volatility = float(request.GET.get("volatility"))
    rate = float(request.GET.get("rate"))
    maturity = float(request.GET.get("maturity"))
    strike_price = float(request.GET.get("strike_price"))
    option_type = float(request.GET.get("option_type"))
    N = float(request.GET.get("observation_num"))
    path_num = int(request.GET.get("path_num"))
    quasi = bool(request.GET.get("quasi")=='true')  #True or False
    # price, std, confcv = premium_project.GPU_european_option(strike_price, maturity, rate, volatility, spot_price, N,
    #                                                          option_type, path_num, quasi)
    price, std, confcv = premium_project_oop.GPU_european_option(strike_price, maturity, rate, volatility, spot_price, N,
                                                             option_type, path_num, quasi)
    return HttpResponse(str(price) + ':' + str(confcv))


def get_GPU_arithmetic_basket_option(request):
    S1 = float(request.GET.get("S1"))
    S2 = float(request.GET.get("S2"))
    V1 = float(request.GET.get("V1"))
    V2 = float(request.GET.get("V2"))
    R = float(request.GET.get("R"))
    T = float(request.GET.get("T"))
    K = float(request.GET.get("K"))
    geo_K = K
    rou = float(request.GET.get("rou"))
    option_type = float(request.GET.get("option_type"))
    path_num = int(request.GET.get("path_num"))
    control_variate = request.GET.get("control_variate")
    quasi = bool(request.GET.get("quasi")=='true')  #True or False
    # price, std, confcv = premium_project.GPU_arithmetic_basket_option(S1, S2, V1, V2, R, T, K, geo_K, rou, option_type,
    #                                                                   path_num, control_variate, quasi)
    price, std, confcv = premium_project_oop.GPU_arithmetic_basket_option(S1, S2, V1, V2, R, T, K, geo_K, rou, option_type,
                                                                      path_num, control_variate, quasi)
    return HttpResponse(str(price) + ':' + str(confcv))


def get_GPU_arithmetic_asian_option(request):
    S0 = float(request.GET.get("spot_price"))
    V = float(request.GET.get("volatility"))
    R = float(request.GET.get("rate"))
    T = float(request.GET.get("maturity"))
    K = float(request.GET.get("strike_price"))
    geo_K = K
    option_type = float(request.GET.get("option_type"))
    N = float(request.GET.get("observation_num"))
    path_num = int(request.GET.get("path_num"))
    control_variate = request.GET.get("control_variate")
    quasi = bool(request.GET.get("quasi")=='true')  #True or False
    # price, std, confcv = premium_project.GPU_arithmetic_asian_option(K, geo_K, T, R, V, S0, N, option_type, path_num,
    #                                                                  control_variate, quasi)
    price, std, confcv = premium_project_oop.GPU_arithmetic_asian_option(K, geo_K, T, R, V, S0, N, option_type, path_num,
                                                                     control_variate, quasi)
    return HttpResponse(str(price) + ':' + str(confcv))