from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'OptionPricer.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    ('^option_pricer/$', 'V2.views.option_pricer'),
    ('^bs_euro/$','V2.views.bs_euro'),
    ('^get_geometric_asian_option/$','V2.views.get_geometric_asian_option'),
    ('^get_arithmetic_asian_option/$','V2.views.get_arithmetic_asian_option'),
    ('^get_geometric_basket_option/$','V2.views.get_geometric_basket_option'),
    ('^get_arithmetic_basket_option/$','V2.views.get_arithmetic_basket_option'),
    ('^option_pricer_premium/$', 'V2.views.goto_premium'),
    ('^get_standardMC_european_option/$','V2.views.get_standardMC_european_option'),
    ('^get_GPU_european_option/$','V2.views.get_GPU_european_option'),
    ('^get_GPU_arithmetic_basket_option/$','V2.views.get_GPU_arithmetic_basket_option'),
    ('^get_GPU_arithmetic_asian_option/$','V2.views.get_GPU_arithmetic_asian_option'),
)
