import uuid
from django.db import models

class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        return f"Conversation {self.id}"

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages', null=True)
    ROLE_CHOICES = [
        ('user', 'User'),
        ('ai', 'AI'),
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    sql_executed = models.TextField(blank=True, null=True)
    data_headers = models.TextField(blank=True, null=True)
    viz_code = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
