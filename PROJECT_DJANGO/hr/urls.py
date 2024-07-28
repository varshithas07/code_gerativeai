from django.urls import path,re_path
from hr import views


urlpatterns = [

    path('', views.home, name='home'),
    path('athena/', views.athena_chat, name='athena'),
     path('save_email_content/', views.save_email_content, name='save_email_content'),


]