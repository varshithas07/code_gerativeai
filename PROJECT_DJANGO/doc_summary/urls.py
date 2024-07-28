from django.urls import path,re_path
from doc_summary import views
app_name = 'doc_summary'

urlpatterns = [
    path('doc_upload/', views.doc_upload, name='doc_upload'),
    path('process_file/', views.process_file, name='process_file'),
    # path(r'^process_file/$',views.process_file, name='process_file'),
    path('doc_chat/', views.doc_chat, name='doc_chat'),
    path('test/', views.test, name='test'),
]