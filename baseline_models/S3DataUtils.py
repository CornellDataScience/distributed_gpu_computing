'''

# S3DataUtils
This notebook defines a utility method download_files for downloading data from S3 to the local filesystem of the current machine.

If you'd like to use download_files in another notebook, load this notebook via a %run command (see Running a Notebook from Another Notebook).
'''
import os
import tensorflow as tf

class S3DataDownloader:
  """
  Exposes utilities for downloading data from S3 to the local filesystem of the current machine.
  
  This class has a single user-facing entry point: download_files(), which takes in a worker id
  and S3 bucket information, then downloads files for the current worker.
  """
  def __init__(self, use_authentication, region_name):
    """
    Wrapper over Boto functions that facilitates downloading data from S3.

    Args:
      use_authentication: Bool, whether to authenticate on requests to S3. If True, reads AWS credentials from the default provider toolchain 
                       (see http://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/credentials.html#credentials-default)
      region_name: String, AWS region of buckets accessed by this S3DataDownloader.
    """
    import boto3
    resource = boto3.resource('s3', region_name=region_name)
    if not use_authentication:
      from botocore.handlers import disable_signing
      resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    self.s3resource = resource        
    
  def _list_files(self, bucket_name, prefix):
    """ List files with the provided prefix in the specified bucket """
    mybucket = self.s3resource.Bucket(bucket_name)
    keys = [obj.key for obj in mybucket.objects.filter(Prefix=prefix)]
    keys.sort()
    return keys
  
  def _download_file(self, bucket_name, key, local_dir):
    """ Downloads the specified S3 object (identified by bucket + key) into the local directory. """
    # Determine the (local) destination path of the S3 object and download it to that path
    dest_filename = os.path.basename(key)
    dest_path = os.path.join(local_dir, dest_filename)
    tf.logging.info("Copying key %s in bucket %s to %s"%(key, bucket_name, dest_path))
    self.s3resource.Bucket(bucket_name).download_file(key, dest_path)
          
  def download_files(self, worker_num, num_workers, bucket_name, prefix, local_dir):
    """
    Downloads files (objects) from an S3 bucket into a local directory.
    
    Args:
      worker_num: Int, index of the current Tensorflow worker
      num_workers: Int, total number of Tensorflow workers
      bucket_name: String, name of the S3 bucket from which to download objects
      prefix: String, all objects whose keys match the specified prefix will be downloaded
      local_dir: String, destination directory for files downloaded from S3
    """
    if not os.path.exists(local_dir):
      os.makedirs(local_dir)
    # List all files in the S3 bucket matching the specified prefix  
    all_files = self._list_files(bucket_name, prefix)
    # Iterate over file indices of the form (worker_num + k * num_workers), downloading each file
    for i in xrange(worker_num, len(all_files), num_workers):
      self._download_file(bucket_name, all_files[i], local_dir)

