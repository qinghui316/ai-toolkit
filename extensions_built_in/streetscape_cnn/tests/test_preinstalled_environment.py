import unittest

from tools.streetscape_cnn.server_preflight import EXPECTED_PACKAGES, validate_environment_profile


class PreinstalledEnvironmentTests(unittest.TestCase):
    def test_locked_profile_remains_version_strict(self):
        validate_environment_profile('locked', dict(EXPECTED_PACKAGES), '12.8')
        packages = dict(EXPECTED_PACKAGES, torchao='0.10.0')
        with self.assertRaises(ValueError):
            validate_environment_profile('locked', packages, '12.8')
        with self.assertRaises(ValueError):
            validate_environment_profile('locked', dict(EXPECTED_PACKAGES), '13.0')

    def test_preinstalled_records_existing_dependencies_without_reinstall(self):
        packages = dict(EXPECTED_PACKAGES, torch='2.9.1+cu128', torchao='0.10.0',
                        torchcodec='0.15.0', bitsandbytes='0.49.2')
        validate_environment_profile('preinstalled', packages, '12.8')
        self.assertEqual(packages['torchao'], '0.10.0')

    def test_missing_packages_cpu_only_and_unknown_profiles_are_rejected(self):
        packages = dict(EXPECTED_PACKAGES)
        packages.pop('torchao')
        with self.assertRaises(ValueError):
            validate_environment_profile('preinstalled', packages, '12.8')
        for profile, cuda in [('preinstalled', None), ('unregistered', '12.8')]:
            with self.assertRaises(ValueError):
                validate_environment_profile(profile, dict(EXPECTED_PACKAGES), cuda)
