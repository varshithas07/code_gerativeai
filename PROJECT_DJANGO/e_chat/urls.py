from django.urls import path,re_path
from e_chat import views
app_name = 'e_gov_chat'

urlpatterns = [
    path('e_chat/', views.e_chat, name='e_chat'),
    path('e_doc/', views.e_doc, name='e_doc'),
]