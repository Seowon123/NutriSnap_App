from django.db import models


class Nutrient(models.Model):
    foodName = models.CharField(max_length=200)
    kcal = models.FloatField()
    water = models.FloatField()
    protein = models.FloatField()
    fat = models.FloatField()
    asche = models.FloatField()
    carbohydrate = models.FloatField()
    sugar = models.FloatField()
    dietaryFibre = models.FloatField()
    calcium = models.FloatField()
    iron = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    sodium = models.FloatField()
    vitaminA = models.FloatField()
    vitaminC = models.FloatField()
    vitaminD = models.FloatField()
    vitaminB1 = models.FloatField()
    vitaminB2 = models.FloatField()
    vitaminB3 = models.FloatField()
    retinol = models.FloatField()
    betaCarotene = models.FloatField()
    cholesterol = models.FloatField()
    saturatedFattyAcid = models.FloatField()
    transFattyAcid = models.FloatField()
    create_date = models.DateTimeField()

    def __str__(self):
        return self.foodName


class Person(models.Model):
    name = models.CharField(max_length=200)
    kcal = models.FloatField()
    water = models.FloatField()
    protein = models.FloatField()
    fat = models.FloatField()
    asche = models.FloatField()
    carbohydrate = models.FloatField()
    sugar = models.FloatField()
    dietaryFibre = models.FloatField()
    calcium = models.FloatField()
    iron = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    sodium = models.FloatField()
    vitaminA = models.FloatField()
    vitaminC = models.FloatField()
    vitaminD = models.FloatField()
    vitaminB1 = models.FloatField()
    vitaminB2 = models.FloatField()
    vitaminB3 = models.FloatField()
    retinol = models.FloatField()
    betaCarotene = models.FloatField()
    cholesterol = models.FloatField()
    saturatedFattyAcid = models.FloatField()
    transFattyAcid = models.FloatField()
    create_date = models.DateTimeField()

class Profile(models.Model):
    image = models.ImageField(upload_to='images/')
    result = models.CharField(max_length=100, null=True)

    def __str__(self):
        return str(self.id)
