

class DeprecatedMetaClass(type):
    """
    A meta-class to indicate deprecation, it satisfies the following :
        - Instantiation of a deprecated class raises a warning
        - Subclassing of a deprecated class raises a warning
        - Support isinstance and issubclass checks

    See https://stackoverflow.com/questions/9008444/how-to-warn-about-class-name-deprecation
    """

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        alias = classdict.get('_DeprecatedClassMeta__alias')

        if alias is not None:
            def new(cls, *args, **kwargs):
                alias = getattr(cls, '_DeprecatedClassMeta__alias')

                if alias is not None:
                    warn("{} has been renamed to {}, the alias will be "
                         "removed in the future".format(cls.__name__,
                             alias.__name__), DeprecationWarning, stacklevel=2)

                return alias(*args, **kwargs)

            classdict['__new__'] = new
            classdict['_DeprecatedClassMeta__alias'] = alias

        fixed_bases = []

        for b in bases:
            alias = getattr(b, '_DeprecatedClassMeta__alias', None)

            if alias is not None:
                warn("{} has been renamed to {}, the alias will be "
                     "removed in the future".format(b.__name__,
                         alias.__name__), DeprecationWarning, stacklevel=2)

            # Avoid duplicate base classes.
            b = alias or b
            if b not in fixed_bases:
                fixed_bases.append(b)

        fixed_bases = tuple(fixed_bases)

        return super().__new__(cls, name, fixed_bases, classdict,
                               *args, **kwargs)

    def __instancecheck__(cls, instance):
        return any(cls.__subclasscheck__(c)
            for c in {type(instance), instance.__class__})

    def __subclasscheck__(cls, subclass):
        if subclass is cls:
            return True
        else:
            return issubclass(subclass, getattr(cls,
                              '_DeprecatedClassMeta__alias'))