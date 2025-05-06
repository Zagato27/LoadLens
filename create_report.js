import React, { useState } from 'react';
import { TextField, Button, Select, MenuItem, CircularProgress, Snackbar } from '@mui/material';

const ReportForm = () => {
  const [params, setParams] = useState({
    start: '',
    end: '',
    user: '',
    password: '',
    url_basic: '',
    space_conf: '',
    page_parent_id: '',
    service: '',
    testType: 'full'
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(false);

  const handleChange = (e) => {
    setParams({ ...params, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(false);
    try {
      const response = await fetch("http://localhost:5000/create_report", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      });
      const result = await response.json();
      setMessage(result.message);
    } catch (error) {
      setMessage(Ошибка создания отчета: ${error.message});
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '500px', margin: 'auto' }}>
      <TextField label="Start Time" name="start" onChange={handleChange} fullWidth margin="normal" />
      <TextField label="End Time" name="end" onChange={handleChange} fullWidth margin="normal" />
      <TextField label="User" name="user" onChange={handleChange} fullWidth margin="normal" />
      <TextField label="Password" type="password" name="password" onChange={handleChange} fullWidth margin="normal" />
      <TextField label="URL Basic" name="url_basic" onChange={handleChange} fullWidth margin="normal" />
      <TextField label="Space Conf" name="space_conf" onChange={handleChange} fullWidth margin="normal" />
      <TextField label="Parent Page ID" name="page_parent_id" onChange={handleChange} fullWidth margin="normal" />
      <TextField label="Service" name="service" onChange={handleChange} fullWidth margin="normal" />
      <Select
        label="Test Type"
        name="testType"
        value={params.testType}
        onChange={handleChange}
        fullWidth
        margin="normal"
      >
        <MenuItem value="Kafka">Kafka</MenuItem>
        <MenuItem value="configuration">Configuration</MenuItem>
        <MenuItem value="default">Default</MenuItem>
        <MenuItem value="full">Full</MenuItem>
      </Select>
      <Button
        variant="contained"
        color="primary"
        onClick={handleSubmit}
        disabled={loading}
        fullWidth
        style={{ marginTop: '20px' }}
      >
        {loading ? <CircularProgress size={24} /> : 'Создать отчет'}
      </Button>
      <Snackbar
        open={Boolean(message)}
        autoHideDuration={6000}
        onClose={() => setMessage('')}
        message={message}
        severity={error ? 'error' : 'success'}
      />
    </div>
  );
};

export default ReportForm;