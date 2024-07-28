from django.urls import path,re_path
from django.contrib import admin
from doc1 import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
 path('doc/', views.doc_view, name='doc'),
  # URL pattern for the doc_view view
path('process_file/', views.process_file, name='process_file'),
path('doc_chat/', views.doc_chat, name='doc_chat'),
path('doc_chat/', views.doc_chat, name='doc_chat')
    # path('get_user_input/', views.get_user_input, name='get_user_input'),


]
# path('split_pdf/', views.split_pdf, name='split_pdf'),]