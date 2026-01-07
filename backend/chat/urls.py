from django.urls import path
from . import views

urlpatterns = [
    path('api/chat', views.chat_view, name='chat'),
    path('auth/login', views.login_view, name='login'),
    path('auth/logout', views.logout_view, name='logout'),
    path('auth/check', views.check_auth_view, name='check_auth'),
]
