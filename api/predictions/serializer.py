from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    home_team_rank = serializers.FloatField()
    away_team_rank = serializers.FloatField()
    home_recent_wins = serializers.FloatField()
    away_recent_wins = serializers.FloatField()
    home_goals_scored_avg = serializers.FloatField()
    away_goals_scored_avg = serializers.FloatField()
