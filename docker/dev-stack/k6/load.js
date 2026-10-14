// Stepped arrival-rate load on three services of the dev stack: Keycloak (JVM, Micrometer),
// the FastAPI app (Python) and the Prometheus HTTP API (Go). Some requests fail on purpose so the
// error queries of the packs have data. K6_DURATION_SCALE shortens the stages (k6-native uses 0.5).
import http from 'k6/http';
import { check, group } from 'k6';

const scale = Number(__ENV.K6_DURATION_SCALE || '1');
const stage = (minutes) => `${Math.max(1, Math.round(minutes * 60 * scale))}s`;

export const options = {
  discardResponseBodies: true,
  scenarios: {
    steps: {
      executor: 'ramping-arrival-rate',
      startRate: 5,
      timeUnit: '1s',
      preAllocatedVUs: 20,
      maxVUs: 120,
      stages: [
        { target: 20, duration: stage(0.5) },
        { target: 20, duration: stage(2) },
        { target: 40, duration: stage(0.5) },
        { target: 40, duration: stage(2) },
        { target: 60, duration: stage(0.5) },
        { target: 60, duration: stage(2) },
      ],
    },
  },
};

const KEYCLOAK = 'http://keycloak:8080';
const PYAPP = 'http://pyapp:8000';
const PROMETHEUS = 'http://prometheus:9090';

export default function () {
  const pick = Math.random();
  if (pick < 0.45) {
    group('keycloak', () => {
      const res = http.get(`${KEYCLOAK}/realms/master/.well-known/openid-configuration`, { tags: { name: 'GET openid-configuration' } });
      check(res, { 'status is 200': (r) => r.status === 200 });
    });
  } else if (pick < 0.85) {
    group('pyapp', () => {
      const id = Math.floor(Math.random() * 1000);
      const res = http.get(`${PYAPP}/items/${id}`, { tags: { name: 'GET /items/{id}' } });
      check(res, { 'status is 200': (r) => r.status === 200 });
    });
  } else if (pick < 0.97) {
    group('prometheus', () => {
      const res = http.get(`${PROMETHEUS}/api/v1/query?query=up`, { tags: { name: 'GET /api/v1/query' } });
      check(res, { 'status is 200': (r) => r.status === 200 });
    });
  } else {
    group('errors', () => {
      const res = http.get(`${PYAPP}/fail`, { tags: { name: 'GET /fail' } });
      check(res, { 'status is 200': (r) => r.status === 200 });
    });
  }
}
