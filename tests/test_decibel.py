from acoustic_toolbox.decibel import dbsum, dbmean, dbadd, dbsub, dbmul, dbdiv


class TestDecibel:
    """Test :mod:`acoustic_toolbox.decibel`"""

    def test_dbsum(self):
        assert abs(dbsum([10.0, 10.0]) - 13.0103) < 1e-5

    def test_dbmean(self):
        assert dbmean([10.0, 10.0]) == 10.0

    def test_dbadd(self):
        assert abs(dbadd(10.0, 10.0) - 13.0103) < 1e-5

    def test_dbsub(self):
        assert abs(dbsub(13.0103, 10.0) - 10.0) < 1e-5

    def test_dbmul(self):
        assert abs(dbmul(10.0, 2) - 13.0103) < 1e-5

    def test_dbdiv(self):
        assert abs(dbdiv(13.0103, 2) - 10.0) < 1e-5
