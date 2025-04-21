from django.db import models
from django.utils.timezone import now

class StudentInformation(models.Model):
    name = models.TextField()
    exam_roll_no = models.IntegerField()
    level = models.TextField()
    student_id = models.TextField(primary_key=True, editable=True)
    campus = models.TextField()
    programme = models.TextField()
    created_at = models.DateTimeField(default=now)

    class Meta:
        ordering = ['-created_at']

class SubjectInformation(models.Model):
    subject_code = models.TextField()
    title = models.TextField(primary_key=True)
    full_marks_ass = models.IntegerField()
    full_marks_final = models.IntegerField()
    pass_marks_ass = models.IntegerField()
    pass_marks_final = models.IntegerField()


class MarksObtained(models.Model):
    student_id = models.ForeignKey(StudentInformation, on_delete=models.CASCADE)
    title = models.ForeignKey(SubjectInformation, on_delete=models.CASCADE)
    year_part = models.TextField()
    marks_assessment = models.TextField()
    marks_final = models.TextField()
    total_marks = models.TextField()
