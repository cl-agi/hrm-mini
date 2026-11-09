from hrm.models.hrm import HRM, HRMConfig

class TRM(HRM):
    def make_low_level(self, config):
        return self.H_level
