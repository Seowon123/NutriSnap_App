from django.db import models

# Create your models here.

class Info(models.Model):
    subject = models.CharField(max_length=200)

    def __str__(self):
        return self.subject


class Person(models.Model):
    name = models.CharField(max_length=200)
    bmi = models.FloatField()
    arginine = models.FloatField()
    tyrosine = models.FloatField()
    CoenzymeQ10 = models.FloatField()
    betaine = models.FloatField()
    BloodPressure = models.FloatField()
    BloodSugar = models.FloatField()
    magnesium = models.FloatField()
    calcium = models.FloatField()
    BoneMass = models.FloatField()
    iron = models.FloatField()
    zinc = models.FloatField()
    vitaminC = models.FloatField()
    vitaminD = models.FloatField()
    cholesterol = models.FloatField()
    NeutralFat = models.FloatField()
    obesity = models.FloatField()
    BodyFat = models.FloatField()
    create_date = models.DateTimeField()

class Profile(models.Model):
    image = models.ImageField(upload_to='images/')
    result = models.CharField(max_length=100, null=True)

    def __str__(self):
        return str(self.id)