class AuthRouter:
    """
    Routes auth-related models to the shared CRM database.
    All other models use the default database.
    """
    
    # Django's built-in auth models
    auth_app_labels = {'auth', 'contenttypes', 'sessions', 'admin'}
    
    def db_for_read(self, model, **hints):
        if model._meta.app_label in self.auth_app_labels:
            return 'auth_db'
        return 'default'
    
    def db_for_write(self, model, **hints):
        if model._meta.app_label in self.auth_app_labels:
            return 'auth_db'
        return 'default'
    
    def allow_relation(self, obj1, obj2, **hints):
        # Allow relations within the same database
        db1 = 'auth_db' if obj1._meta.app_label in self.auth_app_labels else 'default'
        db2 = 'auth_db' if obj2._meta.app_label in self.auth_app_labels else 'default'
        return db1 == db2
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        # Auth models only migrate to auth_db
        if app_label in self.auth_app_labels:
            return db == 'auth_db'
        # All other models only migrate to default
        return db == 'default'
